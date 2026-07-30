"""
Browser environment factory.

Centralizes "produce a ready ``env_config`` and a live ``env``" logic for the
browser example, separate from the generation / tokenization code in
``generate_browser.py``. The mode-specific constructors themselves live next to
this file (``sandbox_env``, ``browser_use_env``, ``remote_env``, ``web_env``);
``create_env`` is the dispatcher across them.
"""

import asyncio
import contextlib
import json
import os

import yaml

from examples.orchard_gui.adapters.browser_adapter import BrowserEnvConfig
from examples.orchard_gui.env.web_env import WebEnv

# ``examples/orchard_gui/`` (parent of this ``env/`` directory). Local resource
# paths in the config (tool list, policy) are resolved relative to it.
_BROWSER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENV_DIR = os.path.dirname(os.path.abspath(__file__))

# One-time-per-process cleanup of leftover sandboxes / Browser-Use sessions from
# a previous aborted run. Kept module-level so the "first creation sweeps" the
# semantics survives across every ``create_env`` call in the process.
_SANDBOX_CLEANUP_DONE = False
_BROWSER_USE_CLEANUP_DONE = False

# Process-wide cap on concurrent env creation (all modes). Created lazily on the
# first ``create_env`` call so it binds to the running event loop.
_ENV_CREATE_SEMAPHORE: "asyncio.Semaphore | None" = None


def _get_env_create_semaphore(env_config: BrowserEnvConfig) -> "asyncio.Semaphore | None":
    """Return a process-wide semaphore bounding concurrent env creation, or None.

    Every mode has a burst-prone creation step — sandbox pod-create (which can
    overwhelm the K8s API server), Browser-Use cloud-session create, even local
    Chromium launch. A rollout batch fires many ``create_env`` calls at once, so
    this caps how many provision simultaneously. Sized from
    ``env_config['max_env_concurrency']``; a value <= 0 or absent disables the limit.
    """
    global _ENV_CREATE_SEMAPHORE
    raw = env_config.get("max_env_concurrency")
    limit = int(raw) if raw is not None else 0
    if limit <= 0:
        return None
    if _ENV_CREATE_SEMAPHORE is None:
        _ENV_CREATE_SEMAPHORE = asyncio.Semaphore(limit)
    return _ENV_CREATE_SEMAPHORE


def load_env_config(env_mode: str | None = None) -> BrowserEnvConfig:
    """Load ``env/config.yaml``, optionally overriding the mode."""
    env_config_path = os.path.join(_ENV_DIR, "config.yaml")
    if not os.path.exists(env_config_path):
        raise ValueError(f"❗  Env config path {env_config_path} does not exist.")
    with open(env_config_path, "r") as f:
        env_config = yaml.safe_load(f)

    if env_mode:
        env_config["mode"] = env_mode

    return env_config


def load_local_resources(env_config: BrowserEnvConfig):
    """Load tool_list and policy from the local filesystem."""
    tool_list = []
    if env_config.get("path_to_tool_list"):
        _tool_list_path = os.path.join(_BROWSER_DIR, env_config["path_to_tool_list"])
        if os.path.exists(_tool_list_path):
            with open(_tool_list_path, "r") as f:
                tool_list = json.load(f)

    policy = ""
    if env_config.get("path_to_policy"):
        _policy_path = os.path.join(_BROWSER_DIR, env_config["path_to_policy"])
        if os.path.exists(_policy_path):
            with open(_policy_path, "r") as f:
                policy = f.read()

    return tool_list, policy


async def create_env(task_data: dict, env_config: BrowserEnvConfig):
    """Create a browser gym environment, bounded by an optional process-wide
    concurrency limit (``max_env_concurrency``) so a rollout batch doesn't fire a
    thundering herd of env creations. See ``_create_env_dispatch`` for the
    per-mode construction.
    """
    sem = _get_env_create_semaphore(env_config)
    async with (sem or contextlib.nullcontext()):
        return await _create_env_dispatch(task_data, env_config)


async def _create_env_dispatch(task_data: dict, env_config: BrowserEnvConfig):
    """
    Create browser gym environment.

    Supports these modes (determined by ``env_config['mode']``):

    - ``"sandbox"``:     K8s sandbox pods via orchestrator + exec/curl
    - ``"remote"``:      connect to pre-started env_server(s) via HTTP
    - ``"local"``:       create WebEnv directly (no server)
    - ``"browser-use"``: drive a remote browser on cloud.browser-use.com via CDP

    Tool list and policy are loaded from the local filesystem.
    Task data is passed in directly (from sample.metadata).

    Returns:
        env instance
    """
    mode = env_config.get("mode", "local")
    tool_list, policy = load_local_resources(env_config)
    task_id = task_data.get("task_id", "unknown")

    # ---- Browser-Use mode: create a fresh remote browser session ----
    if mode == "browser-use":
        from examples.orchard_gui.env.clients.browser_use_env import create_browser_use_env, cleanup_existing_browser_use_sessions

        browser_use_cfg = env_config.get("browser_use", {}) or {}

        # On first session creation, stop any leftover remote sessions
        # from a previous run that was aborted (e.g. Ctrl+C). Remote
        # Browser-Use sessions bill by the minute, so this sweep avoids
        # unexpected charges until the server-side timeout kicks in.
        global _BROWSER_USE_CLEANUP_DONE
        if not _BROWSER_USE_CLEANUP_DONE:
            await cleanup_existing_browser_use_sessions(browser_use_cfg)
            _BROWSER_USE_CLEANUP_DONE = True

        env = await create_browser_use_env(
            browser_use_cfg=browser_use_cfg,
            env_config=env_config,
            tool_list=tool_list,
            policy=policy,
            start_url=task_data["start_url"],
        )
        return env

    # ---- Sandbox mode: create a fresh sandbox for this generation ----
    if mode == "sandbox":
        sandbox_cfg = env_config.get("sandbox", {})

        from examples.orchard_gui.env.clients.sandbox_env import create_sandbox_env, cleanup_existing_sandboxes

        # On first sandbox creation, clean up any leftover sandboxes
        # from a previous run that was aborted (e.g. Ctrl+C).
        global _SANDBOX_CLEANUP_DONE
        if not _SANDBOX_CLEANUP_DONE:
            await cleanup_existing_sandboxes(sandbox_cfg)
            _SANDBOX_CLEANUP_DONE = True

        env = await create_sandbox_env(sandbox_cfg)
        await env.initialize(
            task_id=task_id,
            task_data=task_data,
            env_config=env_config,
            tool_list=tool_list,
            policy=policy,
        )

        return env

    # ---- Remote mode: borrow a slot on a pre-started env_server ----
    if mode == "remote":
        from examples.orchard_gui.env.clients.remote_env import get_env_pool

        server_urls = (env_config.get("remote", {}) or {}).get("server_url", "")
        if not server_urls:
            raise ValueError("Set remote.server_url in env/config.yaml")

        pool = await get_env_pool(server_urls)
        server_url, env_id = await pool.acquire()
        env = pool.create_remote_env(server_url, env_id)
        try:
            await env.initialize(
                task_id=task_id,
                task_data=task_data,
                env_config=env_config,
                tool_list=tool_list,
                policy=policy,
            )
        except BaseException:
            # create_env's caller only cleans up via env.exit() once the env
            # is returned; release the slot here so a failed init never leaks it.
            pool.release(server_url, env_id)
            raise

        return env

    # ---- Local mode: create WebEnv directly ----
    env = WebEnv(
        width=env_config["width"],
        height=env_config["height"],
        dpr=env_config["dpr"],
        max_retries=env_config["max_retries"],
        wait_timeout=env_config["wait_timeout"],
        start_url=task_data["start_url"],
        resize_output_coords=env_config["resize_output_coords"],
        resize_scale=env_config["resize_scale"],
        image_patch_size=env_config["image_patch_size"],
        tool_list=tool_list,
        policy=policy
    )

    # Initialize async resources
    await env.setup()

    return env
