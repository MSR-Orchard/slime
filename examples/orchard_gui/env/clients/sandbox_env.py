"""
Sandbox-based Browser Environment Client

Provides ``SandboxWebEnv`` -- a drop-in replacement for ``WebEnv`` that
communicates with a K8s sandbox pod through the sandbox orchestrator's
``exec()`` + ``curl`` commands.

Each generation creates a fresh sandbox via ``create_sandbox_env()`` and
deletes it when ``env.exit()`` is called.
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import the sandbox client from the orchard_env repo, which ships alongside
# this tree at orchard/orchard_env/ (six levels up from env/clients/). Its
# repo root goes on sys.path so the orchard_env package inside it resolves —
# no symlink or vendored copy needed. A pip-installed orchard_env works too
# (the sibling repo, when present, takes precedence).
# ---------------------------------------------------------------------------
_ORCHARD_ENV_REPO = os.path.abspath(
    os.path.join(os.path.dirname(__file__), *[os.pardir] * 6, "orchard_env")
)
if os.path.isdir(_ORCHARD_ENV_REPO) and _ORCHARD_ENV_REPO not in sys.path:
    sys.path.insert(0, _ORCHARD_ENV_REPO)

try:
    from orchard_env.client.sandbox_client import AsyncSandboxClient, AsyncSandboxInstance
except ImportError as exc:
    raise ImportError(
        f"orchard_env is not importable (no repo at {_ORCHARD_ENV_REPO} and not "
        "installed in this environment); clone it alongside trainer/ or pip install it"
    ) from exc

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_ENV_SERVER_PORT = 8100

# In-sandbox destination for the injected env-server code (see
# _build_env_code_payload). /tmp is writable in every image variant.
_ENV_INJECT_DIR = "/tmp/envsrv"

_env_code_payload_cache: "tuple[bytes, str] | None" = None


def _build_env_code_payload() -> "tuple[bytes, str]":
    """Package the repo's env-server code for injection into a sandbox.

    Returns ``(tgz_bytes, dotted_module)`` where the tarball recreates the
    repo's package layout (examples/<pkg>/env/...) and ``dotted_module`` is the
    matching ``python -m`` target for env_server. Everything is derived from
    this file's own location, so repo renames propagate automatically — the
    sandbox image only needs the base deps (python + fastapi/uvicorn +
    playwright/Chromium), never a copy of this code.
    """
    global _env_code_payload_cache
    if _env_code_payload_cache is not None:
        return _env_code_payload_cache

    env_dir = Path(__file__).resolve().parent.parent  # .../examples/<pkg>/env
    parts = env_dir.parts
    idx = parts.index("examples")
    pkg_prefix = parts[idx:]  # ("examples", "<pkg>", "env")
    dotted_module = ".".join(pkg_prefix) + ".docker_server.env_server"

    # Keep in sync with what env_server imports (see docker_server/env_server.py
    # and web_env.py); prompts/tasks/config are not needed server-side.
    manifest = [
        "base_env.py",
        "web_env.py",
        "utils/js.py",
        "utils/keyboard.py",
        "docker_server/env_server.py",
    ]

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        # __init__.py stubs for every package level
        init_dirs = [
            Path(*pkg_prefix[: i + 1]) for i in range(len(pkg_prefix))
        ] + [Path(*pkg_prefix) / "utils", Path(*pkg_prefix) / "docker_server"]
        for d in init_dirs:
            info = tarfile.TarInfo(str(d / "__init__.py"))
            info.size = 0
            tar.addfile(info, io.BytesIO(b""))
        for rel in manifest:
            data = (env_dir / rel).read_bytes()
            info = tarfile.TarInfo(str(Path(*pkg_prefix) / rel))
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

    _env_code_payload_cache = (buf.getvalue(), dotted_module)
    return _env_code_payload_cache
# Backstop on the sandbox exec/curl. For /step and /exit the generate-side
# asyncio guards (env_step_timeout=120, env_exit_timeout=60) are shorter and fire
# first; this only bounds the longer init path. Keep it >= those guards.
_CURL_TIMEOUT = 300  # seconds for curl requests
_EXEC_CURL_MAX_RETRIES = 3  # retries for exec/curl failures inside sandbox
_EXEC_CURL_RETRY_DELAY = 2  # initial backoff delay in seconds

_ENV_SERVER_STARTUP_TIMEOUT = 120  # seconds
_ENV_SERVER_STARTUP_POLL = 3  # seconds between health checks

# Budget for the sandbox pod to become ready (create_sandbox(timeout=...)).
# The orchestrator no longer takes a per-sandbox lease here: sandbox lifetime
# is now bounded server-side by a TTL plus heartbeat-based cleanup (a sandbox
# that has sent at least one heartbeat is reaped a few minutes after beats
# stop). Generous so pods survive node autoscaling / cold image pulls.
_SANDBOX_CREATE_TIMEOUT = 900  # seconds

# Interval of the client keep-alive heartbeat. Must stay well under the
# orchestrator's heartbeat timeout (~180s) or live sandboxes get reaped.
_HEARTBEAT_INTERVAL = 60  # seconds


def _build_curl_cmd(method: str, path: str, payload: Optional[dict] = None) -> str:
    """Build a curl command that avoids shell quoting issues.

    Uses base64-encoded stdin for POST payloads so that no special
    characters appear in the shell command string.
    """
    url = f"http://localhost:{_ENV_SERVER_PORT}{path}"

    if method == "GET":
        return f'curl -sf "{url}"'

    # POST: pipe base64-decoded JSON through stdin
    if payload is not None:
        payload_json = json.dumps(payload)
        b64 = base64.b64encode(payload_json.encode()).decode()
        return (
            f'echo {b64} | base64 -d | '
            f'curl -s -X POST "{url}" '
            f'-H "Content-Type: application/json" -d @-'
        )
    else:
        return (
            f'curl -s -X POST "{url}" '
            f'-H "Content-Type: application/json" -d "{{}}"'
        )


class SandboxWebEnv:
    """
    Drop-in replacement for WebEnv / RemoteWebEnv that communicates with
    a sandbox pod running env_server.

    Communication uses the sandbox ``exec()`` method to run ``curl``
    commands inside the pod, hitting ``localhost:8100``.

    Lifecycle:
        1. create_sandbox_env() -- creates sandbox, starts env_server, returns this env
        2. initialize(task_id) -- calls /reset on the env_server
        3. reset()    -- returns cached observation from initialize()
        4. step(actions) -- calls /step
        5. exit()     -- calls /exit, deletes the sandbox, closes the client
    """

    def __init__(
        self,
        sandbox: AsyncSandboxInstance,
    ) -> None:
        self.sandbox = sandbox

        self._cached_reset: Optional[Tuple[dict, dict]] = None
        self._task_data: Optional[dict] = None

    # ------------------------------------------------------------------
    # Public interface (mirrors WebEnv / RemoteWebEnv)
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        """No-op -- sandbox and env_server are started by create_sandbox_env()."""
        pass

    async def initialize(
        self,
        task_id: str,
        task_data: dict,
        env_config: dict,
        tool_list: list | None = None,
        policy: str = "",
    ) -> None:
        """Reset env with a task via curl /reset. Caches observation for reset().

        All configuration is sent from the client — the server does not
        load any local config/task files.
        """
        payload: Dict[str, Any] = {
            "task_id": task_id,
            "task_data": task_data,
            "env_config": env_config,
            "tool_list": tool_list or [],
            "policy": policy,
        }

        try:
            data = await self._exec_curl("/reset", payload)
        except Exception as exc:
            logger.error("❌ Failed to initialize environment with task_id %s: %s", task_id, exc)
            await self.exit()
            raise

        obs = data["observation"]
        obs["screenshot"] = self._decode_screenshot(obs.get("screenshot"))

        self._cached_reset = (obs, data["info"])
        self._task_data = task_data

    async def reset(
        self,
        url: Optional[str] = None,
        auth_info: Optional[dict] = None,
    ) -> Tuple[dict, dict]:
        """Return the cached observation from initialize()."""
        if self._cached_reset is not None:
            result = self._cached_reset
            self._cached_reset = None
            return result
        raise RuntimeError(
            "❌ SandboxWebEnv.reset() called without a prior initialize(). "
            "Call initialize(task_id) first."
        )

    async def step(
        self, action_list: List[Dict[str, Any]]
    ) -> Tuple[dict, float, bool, bool, dict]:
        """Execute actions via curl /step."""
        try:
            data = await self._exec_curl("/step", {"actions": action_list, "env_id": 0})
        except Exception as exc:
            logger.error("❌ Step failed: %s", exc)
            await self.exit()
            raise

        obs = data["observation"]
        if obs.get("screenshot") is None:
            print("========== Error: Action execution failed without a screenshot. ==========")
            print("Environment message:", data["info"]["env_message"])
            print("Diagnostics:", data.get("diagnostics"))
            await self.exit()
            raise ValueError("Action execution failed without a screenshot.")

        obs["screenshot"] = self._decode_screenshot(obs.get("screenshot"))

        info = data["info"]
        if "diagnostics" in data:
            info["server_diagnostics"] = data["diagnostics"]

        return (
            obs,
            data["reward"],
            data["terminated"],
            data["truncated"],
            info,
        )

    async def exit(self) -> None:
        """Close the browser env via curl /exit, delete the sandbox, close the client."""
        sid = self.sandbox.sandbox_id
        try:
            await self._exec_curl("/exit", {"env_id": 0})
        except Exception as exc:
            logger.warning("❌ SandboxWebEnv env/exit error (ignored): %s", exc)

        try:
            await self.sandbox.delete()
            print(f"♻️  [Deleted {sid}]")
            _remove_sandbox_id(sid)
        except Exception as exc:
            # A 404 / "already deleted" means the orchestrator already reaped the
            # sandbox (e.g. its lease expired), so drop the now-stale marker too.
            # Keep the marker only for a genuine failure (a still-live sandbox we
            # could not delete) so a later cleanup sweep can retry it.
            if _is_already_gone(exc):
                _remove_sandbox_id(sid)
                logger.info("Sandbox %s already gone (%s); removed stale marker.", sid, exc)
            else:
                logger.warning("❌ Failed to delete sandbox %s: %s", sid, exc)

        try:
            await self.sandbox._client.close()
        except Exception as exc:
            logger.warning("❌ Failed to close client session: %s", exc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _exec_curl(self, path: str, payload: dict) -> dict:
        """Execute a curl POST command inside the sandbox and parse the JSON response.

        Retries on transient exec/curl failures (exit_code=None, empty stdout)
        before raising.
        """
        cmd = _build_curl_cmd("POST", path, payload)
        sid = self.sandbox.sandbox_id
        last_error = None

        for attempt in range(_EXEC_CURL_MAX_RETRIES):
            t0 = asyncio.get_event_loop().time()
            result = await self.sandbox.exec(
                cmd,
                timeout=_CURL_TIMEOUT,
                cwd="/app",
                login_shell=False,
            )
            client_elapsed = round(asyncio.get_event_loop().time() - t0, 3)

            if not result.succeeded:
                last_error = (
                    f"⁉️ curl {path} failed in sandbox {sid} "
                    f"(client_elapsed={client_elapsed}s, timeout={_CURL_TIMEOUT}s): "
                    f"exit_code={result.exit_code}, "
                    f"stderr={result.stderr!r}, "
                    f"stdout(first 500)={result.stdout[:500]!r}"
                )
                if attempt < _EXEC_CURL_MAX_RETRIES - 1:
                    wait_time = _EXEC_CURL_RETRY_DELAY * (2 ** attempt)
                    logger.warning(
                        "[exec_curl retry %d/%d] %s — waiting %ds",
                        attempt + 1, _EXEC_CURL_MAX_RETRIES, last_error, wait_time,
                    )
                    await asyncio.sleep(wait_time)
                    continue
                raise RuntimeError(last_error)

            stdout = result.stdout.strip()
            if not stdout:
                last_error = (
                    f"⁉️ curl {path} returned empty response in sandbox {sid} "
                    f"(client_elapsed={client_elapsed}s)"
                )
                if attempt < _EXEC_CURL_MAX_RETRIES - 1:
                    wait_time = _EXEC_CURL_RETRY_DELAY * (2 ** attempt)
                    logger.warning(
                        "[exec_curl retry %d/%d] %s — waiting %ds",
                        attempt + 1, _EXEC_CURL_MAX_RETRIES, last_error, wait_time,
                    )
                    await asyncio.sleep(wait_time)
                    continue
                raise RuntimeError(last_error)

            # Success — break out of retry loop
            break

        try:
            data = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"⁉️ curl {path} returned invalid JSON in sandbox "
                f"{sid}: {exc}\n"
                f"stdout: {stdout}"
            )

        # FastAPI HTTPException returns {"detail": "..."} on errors.
        # curl -s doesn't fail on HTTP 4xx/5xx, so catch it here.
        if "detail" in data:
            detail = data["detail"]
            # The /step endpoint now sends structured detail with diagnostics.
            if isinstance(detail, dict):
                diag = detail.get("diagnostics", {})
                logger.error(
                    "⁉️ env_server %s error in sandbox %s: %s | "
                    "server_diagnostics=%s",
                    path, sid, detail.get("message", detail), diag,
                )
                raise RuntimeError(
                    f"⁉️ env_server {path} error in sandbox {sid}: "
                    f"{detail.get('message', detail)} | "
                    f"server_elapsed={diag.get('elapsed_secs')}s, "
                    f"rss_mb={diag.get('rss_mb')}, "
                    f"server_error={diag.get('error')}, "
                    f"server_traceback={diag.get('error_traceback', '')[:1000]}"
                )
            raise RuntimeError(
                f"⁉️ env_server {path} error in sandbox {sid}: {detail}"
            )

        # Log diagnostics from successful responses
        diag = data.get("diagnostics")
        if diag:
            logger.info(
                "sandbox %s %s OK: server_elapsed=%.3fs, rss_mb=%s, "
                "step_count=%s, uptime=%ss",
                sid, path,
                diag.get("elapsed_secs", -1),
                diag.get("rss_mb"),
                diag.get("request_counts", {}).get("step"),
                diag.get("server_uptime_secs"),
            )

        return data

    @staticmethod
    def _decode_screenshot(value: Any) -> bytes:
        """Decode a base64-encoded screenshot back to bytes."""
        if isinstance(value, str):
            return base64.b64decode(value)
        if isinstance(value, (bytes, bytearray)):
            return bytes(value)
        return b""


# ---------------------------------------------------------------------------
# Public factory function (creates a sandbox, starts env_server, returns env)
# ---------------------------------------------------------------------------


_SANDBOX_MANIFEST_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".sandboxes"
)
print("Sandbox manifest directory:", _SANDBOX_MANIFEST_DIR)


def _save_sandbox_id(sandbox_id: str) -> None:
    """Record a sandbox ID by creating a marker file.

    Each sandbox gets its own file so parallel workers never contend
    on the same file.
    """
    os.makedirs(_SANDBOX_MANIFEST_DIR, exist_ok=True)
    marker = os.path.join(_SANDBOX_MANIFEST_DIR, sandbox_id)
    with open(marker, "w") as f:
        f.write("")


def _remove_sandbox_id(sandbox_id: str) -> None:
    """Remove the marker file for a sandbox ID."""
    marker = os.path.join(_SANDBOX_MANIFEST_DIR, sandbox_id)
    try:
        os.remove(marker)
    except OSError:
        pass


def _is_already_gone(exc: BaseException) -> bool:
    """True if a delete failure means the sandbox no longer exists on the
    orchestrator (HTTP 404 / "not found" / "has been deleted") — e.g. its lease
    already expired — as opposed to a transient failure deleting a live sandbox.

    Used so a delete that 404s still drops the local marker, instead of leaving
    a stale marker for a sandbox that is already gone.
    """
    status = getattr(exc, "status", None) or getattr(exc, "status_code", None)
    if status == 404:
        return True
    msg = str(exc).lower()
    return "404" in msg or "not found" in msg or "has been deleted" in msg


def _list_sandbox_ids() -> List[str]:
    """Return all sandbox IDs recorded in the manifest directory."""
    if not os.path.isdir(_SANDBOX_MANIFEST_DIR):
        return []
    try:
        return os.listdir(_SANDBOX_MANIFEST_DIR)
    except OSError:
        return []


async def cleanup_existing_sandboxes(sandbox_cfg: Dict[str, Any]) -> None:
    """Delete leftover sandboxes recorded in the local manifest directory.

    Call this at startup to clean up sandboxes from a previous run that
    was aborted (e.g. via Ctrl+C).  Reads sandbox IDs from per-sandbox
    marker files and deletes them via the orchestrator API.

    Args:
        sandbox_cfg: The ``sandbox:`` section from config.yaml.
    """
    sandbox_ids = _list_sandbox_ids()
    if not sandbox_ids:
        logger.info("No leftover sandbox markers found, nothing to clean up.")
        return

    orchestrator_url = (
        os.environ.get("SANDBOX_ORCHESTRATOR_URL")
        or sandbox_cfg.get("orchestrator_url")
    )
    if not orchestrator_url:
        raise ValueError("Set SANDBOX_ORCHESTRATOR_URL env var")

    api_key = os.environ.get("SANDBOX_API_KEY") or sandbox_cfg.get("api_key")

    logger.info(f"Found {len(sandbox_ids)} leftover sandbox(es), deleting...")

    try:
        async with AsyncSandboxClient(
            base_url=orchestrator_url,
            api_key=api_key,
            auto_cleanup=True,
        ) as client:

            for sid in sandbox_ids:
                try:
                    await client.delete_sandbox(sid)
                    logger.info(f"♻️  [Deleted {sid}]")
                    _remove_sandbox_id(sid)
                except Exception as exc:
                    # Already-gone (404) → the marker is stale, so drop it. Keep it
                    # only on a genuine failure so a later sweep can retry.
                    if _is_already_gone(exc):
                        _remove_sandbox_id(sid)
                        logger.info(f"Sandbox {sid} already gone; removed stale marker.")
                    else:
                        logger.warning(f"❌ Failed to delete sandbox {sid}: {exc}")

    except Exception as exc:
        logger.warning(f"❌ Failed to cleanup existing sandboxes: {exc}")


async def create_sandbox_env(sandbox_cfg: Dict[str, Any]) -> SandboxWebEnv:
    """Create a fresh sandbox with env_server and return a ``SandboxWebEnv``.

    The returned env owns its own ``AsyncSandboxClient``.  Calling
    ``env.exit()`` will close the browser env, delete the sandbox pod,
    and close the client session.

    Args:
        sandbox_cfg: The ``sandbox:`` section from config.yaml.
    """
    orchestrator_url = (
        os.environ.get("SANDBOX_ORCHESTRATOR_URL")
        or sandbox_cfg.get("orchestrator_url")
    )
    if not orchestrator_url:
        raise ValueError("Set SANDBOX_ORCHESTRATOR_URL env var or sandbox.orchestrator_url in config.yaml")
    api_key = os.environ.get("SANDBOX_API_KEY") or sandbox_cfg.get("api_key")

    # On success, SandboxWebEnv takes ownership of the sandbox (which
    # internally holds a client reference) and closes it in exit().
    # On failure, we clean up the client here via try/except.
    client = AsyncSandboxClient(
        base_url=orchestrator_url,
        api_key=api_key,
        auto_cleanup=True,
    )

    sandbox = None
    try:
        image = sandbox_cfg.get("image", "qianhuiwu/browser-env-lite:latest")
        cpu = sandbox_cfg.get("cpu")
        memory = sandbox_cfg.get("memory")
        block_network = sandbox_cfg.get("block_network", False)

        start_time = time.monotonic()
        print(f"⏱️  Creating sandbox with image={image}, cpu={cpu}, memory={memory}, block_network={block_network}...")
        sandbox = await client.create_sandbox(
            image=image,
            block_network=block_network,
            cpu=cpu,
            memory=memory,
            timeout=_SANDBOX_CREATE_TIMEOUT,
        )
        sid = sandbox.sandbox_id
        elapsed = time.monotonic() - start_time
        print(f"✅ [Created {sid}] {elapsed:.1f}s - Sandbox starting env_server...")
        _save_sandbox_id(sid)

        # The sandbox orchestrator overrides the image CMD with its own agent,
        # so we launch the startup script explicitly.
        #
        # Two image variants are supported:
        #   - battalion-style image (e.g. battalion7244/browser-env-lite-zyw-v1):
        #     ships /opt/websyn_start.sh which starts 15 Flask mirror sites on
        #     :40000-40014, waits for them to be ready, then execs env_server
        #     on :8100 as its final step.
        #   - base image (qianhuiwu/browser-env-lite): no mirror sites, so we
        #     start env_server directly.
        probe = await sandbox.exec(
            "test -x /opt/websyn_start.sh && echo websyn || echo plain",
            timeout=10, cwd="/app", login_shell=False,
        )
        launcher_mode = probe.stdout.strip() if probe.succeeded else "plain"

        if launcher_mode == "websyn":
            # websyn_start.sh internally execs env_server after sites are ready,
            # so the existing /health poll on :8100 remains the correct readiness
            # gate. NOTE: websyn images run their BAKED env_server copy — code
            # injection below does not apply to this branch.
            start_cmd = (
                "nohup /opt/websyn_start.sh "
                "> /tmp/websyn_start.log 2>&1 &"
            )
            start_cwd = "/opt"
        else:
            # Inject the repo's current env-server code instead of relying on a
            # copy baked into the image: the image then only provides the base
            # deps, and env-code changes / repo renames never require an image
            # release. PYTHONPATH + cwd both point at the injected tree, so the
            # image's own /app copy (if any) is never on the import path.
            payload, env_server_module = _build_env_code_payload()
            await sandbox.upload_content(payload, f"{_ENV_INJECT_DIR}.tgz")
            extract = await sandbox.exec(
                f"mkdir -p {_ENV_INJECT_DIR} && tar xzf {_ENV_INJECT_DIR}.tgz -C {_ENV_INJECT_DIR}",
                timeout=15, cwd="/tmp", login_shell=False,
            )
            if not extract.succeeded:
                raise RuntimeError(
                    f"❌ env code injection failed in {sid}: {extract.stderr or extract.stdout}"
                )
            print(f"💉 [Injected {sid}] env code ({len(payload) // 1024} KB, module={env_server_module})")
            start_cmd = (
                f"nohup env PYTHONPATH={_ENV_INJECT_DIR} python -m {env_server_module} "
                f"--host 0.0.0.0 --port {_ENV_SERVER_PORT} "
                "> /tmp/env_server.log 2>&1 &"
            )
            start_cwd = _ENV_INJECT_DIR

        await sandbox.exec(start_cmd, timeout=30, cwd=start_cwd, login_shell=False)

        # Poll health until ready (reset timer — don't count sandbox creation time)
        health_start = time.monotonic()
        health_cmd = f'curl -sf "http://localhost:{_ENV_SERVER_PORT}/health"'

        while time.monotonic() - health_start < _ENV_SERVER_STARTUP_TIMEOUT:
            await asyncio.sleep(_ENV_SERVER_STARTUP_POLL)
            try:
                hresult = await sandbox.exec(
                    health_cmd, timeout=10, cwd="/app", login_shell=False
                )
                if hresult.succeeded and hresult.stdout.strip():
                    # Success — SandboxWebEnv now owns the sandbox
                    # (and its internal client reference).
                    sandbox.start_heartbeat(interval=_HEARTBEAT_INTERVAL)
                    # The orchestrator only applies heartbeat-based cleanup to
                    # sandboxes that have sent >= 1 heartbeat, and
                    # start_heartbeat() first beats after a full interval. Beat
                    # once now so a client killed inside that window leaks the
                    # pod only until the heartbeat timeout, not the multi-hour
                    # TTL. (No public one-shot heartbeat API, hence _request.)
                    try:
                        await client._request(
                            "POST", f"/sandboxes/{sid}/heartbeat"
                        )
                    except Exception:
                        pass  # best effort — the periodic loop takes over
                    return SandboxWebEnv(sandbox=sandbox)
            except Exception:
                pass

        # Timed out — dump diagnostics before cleanup
        try:
            ps_result = await sandbox.exec("ps aux", timeout=10, cwd="/app", login_shell=False)
            print(f"⏱️ [{sid}] Timed out. Processes:\n{ps_result.stdout[:2000]}")
            log_result = await sandbox.exec(
                "for f in /tmp/env_server.log /tmp/websyn_start.log /tmp/websyn_*.log; do "
                "  [ -f \"$f\" ] && { echo \"=== $f ===\"; tail -n 50 \"$f\"; }; "
                "done; echo '---'; ls -la /tmp/websyn_*.log 2>/dev/null || true",
                timeout=10, cwd="/app", login_shell=False,
            )
            print(f"⏱️ [{sid}] Server logs:\n{log_result.stdout[:2000]}")
        except Exception:
            pass
        raise TimeoutError(f"❌ env_server in {sid} not healthy within {_ENV_SERVER_STARTUP_TIMEOUT}s")

    except BaseException:
        # Clean up on any failure
        if sandbox is not None:
            try:
                await sandbox.delete()
                print(f"♻️  [Deleted {sid}]")
                _remove_sandbox_id(sandbox.sandbox_id)
            except Exception as exc:
                raise RuntimeError("❌ Failed to delete sandbox during cleanup") from exc
        try:
            await client.close(cleanup=True)
        except Exception:
            pass
        logger.error("❌ Failed to create sandbox environment", exc_info=True)
        raise


# ---------------------------------------------------------------------------
# CLI entry point: python sandbox_env.py --cleanup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sandbox environment utilities")
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete all leftover sandboxes recorded in .sandboxes/",
    )
    args = parser.parse_args()

    if args.cleanup:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

        sandbox_ids = _list_sandbox_ids()
        if not sandbox_ids:
            print("No sandbox markers found in", _SANDBOX_MANIFEST_DIR)
            sys.exit(0)

        print(f"Found {len(sandbox_ids)} sandbox(es) to clean up:")
        for sid in sandbox_ids:
            print(f"  - {sid}")

        # Build a minimal sandbox_cfg from environment variables
        sandbox_cfg = {
            "orchestrator_url": os.environ.get("SANDBOX_ORCHESTRATOR_URL", ""),
            "api_key": os.environ.get("SANDBOX_API_KEY"),
        }

        asyncio.run(cleanup_existing_sandboxes(sandbox_cfg))
        print("Cleanup complete.")
    else:
        parser.print_help()
