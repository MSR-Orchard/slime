"""
Reward computation for SWE-bench tasks.

This module provides multiple reward strategies based on SWE-bench evaluation:
1. Binary: 1.0 if tests pass (FULL resolution), 0.0 otherwise
2. Multi-objective: Dict with multiple reward components
3. Shaped: Intermediate rewards for progress (optional)

The evaluation follows SWE-bench grading logic:
- Fail-to-Pass (F2P): Tests that should change from failing to passing
- Pass-to-Pass (P2P): Tests that should remain passing (maintenance)
- Resolution: FULL (F2P=1.0 & P2P=1.0), PARTIAL (0<F2P<1 & P2P=1.0), NO (otherwise)
"""

import asyncio
import base64
import json
import logging
import os
import re
import shlex
import subprocess
import tempfile
import traceback
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from slime.utils.types import Sample

from .swe_wrapper import (
    create_environment,
    get_docker_image_name,
    load_config,
    stop_environment,
)

# Import SWE-bench harness components for proper evaluation
try:
    from swebench.harness.test_spec.test_spec import make_test_spec, TestSpec
    from swebench.harness.constants import LATEST, SWEbenchInstance
    from swebench.harness.grading import get_eval_report
    from swebench.harness.constants import (
        KEY_INSTANCE_ID,
        KEY_MODEL,
        KEY_PREDICTION,
        MAP_REPO_VERSION_TO_SPECS,
        MAP_REPO_TO_EXT,
        START_TEST_OUTPUT,
        END_TEST_OUTPUT,
    )
    HAS_SWEBENCH_HARNESS = True
except ImportError:
    HAS_SWEBENCH_HARNESS = False
    make_test_spec = None
    TestSpec = None
    get_eval_report = None
    KEY_INSTANCE_ID = "instance_id"
    KEY_MODEL = "model_name_or_path"
    KEY_PREDICTION = "model_patch"

SWE_TIMEOUT_REWARD_CREATE_ENV = int(os.environ["SWE_TIMEOUT_CREATE_ENV"])
SWE_TIMEOUT_REWARD_EXECUTE = int(os.environ["SWE_TIMEOUT_REWARD_EXECUTE"])
SWE_TIMEOUT_REWARD_TOTAL = int(os.environ["SWE_TIMEOUT_REWARD_TOTAL"])

# ==============================================================================
# Constants and Enums (from SWE-bench)
# ==============================================================================

class ResolvedStatus(Enum):
    """Resolution status for an instance."""
    NO = "RESOLVED_NO"
    PARTIAL = "RESOLVED_PARTIAL"
    FULL = "RESOLVED_FULL"


# Test category constants
FAIL_TO_PASS = "FAIL_TO_PASS"
PASS_TO_PASS = "PASS_TO_PASS"


# ==============================================================================
# Utility Functions
# ==============================================================================

def extract_patch_from_output(text: str) -> str:
    """
    Extract patch/diff from final output.

    Args:
        text: Final output text that may contain a git diff or patch

    Returns:
        Extracted patch string
    """
    # Look for git diff format
    diff_pattern = r"```diff\s*\n(.*?)\n```"
    diff_match = re.search(diff_pattern, text, re.DOTALL)
    if diff_match:
        return diff_match.group(1).strip()

    # Look for patch format
    patch_pattern = r"```patch\s*\n(.*?)\n```"
    patch_match = re.search(patch_pattern, text, re.DOTALL)
    if patch_match:
        return patch_match.group(1).strip()

    # Look for plain code blocks
    code_pattern = r"```\s*\n(.*?)\n```"
    code_match = re.search(code_pattern, text, re.DOTALL)
    if code_match:
        content = code_match.group(1).strip()
        # Check if it looks like a diff
        if content.startswith("diff ") or "@@" in content or content.startswith("---"):
            return content

    # If no code block, check if the whole text looks like a diff
    if text.startswith("diff ") or "@@" in text or text.startswith("---"):
        return text.strip()

    return ""


def is_valid_patch(patch: str) -> bool:
    """
    Check if a patch has valid syntax.

    Args:
        patch: Patch string

    Returns:
        True if patch looks valid
    """
    if not patch:
        return False

    # Check for basic diff markers
    has_diff_markers = any(
        marker in patch
        for marker in ["diff --git", "---", "+++", "@@", "Index:", "==="]
    )

    return has_diff_markers


# ==============================================================================
# Docker Test Execution
# ==============================================================================



async def run_tests_in_docker(
    instance_id: str,
    patch: str,
    repo: str,
    base_commit: str,
    fail_to_pass: list[str] | None = None,
    pass_to_pass: list[str] | None = None,
    instance: dict | None = None,
    config: dict | None = None,
) -> dict:
    """
    Run SWE-bench tests in a Docker container and evaluate results.

    This function follows the same evaluation logic as run_instance_modal() from
    swebench.harness.modal_eval:
    1. Creates a TestSpec from the instance using make_test_spec()
    2. Applies the model's patch using git apply (with fallback to patch command)
    3. Runs the eval_script from the TestSpec
    4. Uses get_eval_report() for proper SWE-bench grading

    Args:
        instance_id: SWE-bench instance ID
        patch: The generated patch to test (model_patch)
        repo: Repository name
        base_commit: Base commit hash
        fail_to_pass: List of tests that should change from failing to passing
        pass_to_pass: List of tests that should remain passing
        instance: Full instance dict for TestSpec creation and Docker image resolution
        config: Optional environment config (loads default if not provided)

    Returns:
        Dict with test results including:
        - passed: bool (True if FULL resolution)
        - resolved: bool (same as passed)
        - resolution: str (FULL, PARTIAL, or NO)
        - fail_to_pass_rate: float
        - pass_to_pass_rate: float
        - tests_run: int
        - error: str
        - output: str
    """
    result = {
        "passed": False,
        "resolved": False,
        "resolution": ResolvedStatus.NO.value,
        "fail_to_pass_rate": 0.0,
        "pass_to_pass_rate": 0.0,
        "tests_run": 0,
        "tests_passed": 0,
        "f2p_total": 0,
        "f2p_passed": 0,
        "p2p_total": 0,
        "p2p_passed": 0,
        "error": "",
        "output": "",
    }

    # Validate patch first
    if not is_valid_patch(patch):
        result["error"] = "Invalid patch format"
        return result

    # Set defaults for test lists
    if fail_to_pass is None:
        fail_to_pass = []
    if pass_to_pass is None:
        pass_to_pass = []

    # Build instance dict for Docker image resolution if not provided
    if instance is None:
        instance = {
            "instance_id": instance_id,
            "repo": repo,
            "base_commit": base_commit,
        }

    # Load config if not provided
    if config is None:
        config_path = os.environ["SWE_CONFIG_PATH"]
        try:
            config = load_config(config_path)
        except Exception as e:
            result["error"] = f"Failed to load config: {e}"
            return result

    # Require SWE-bench harness for evaluation
    if not HAS_SWEBENCH_HARNESS:
        result["error"] = "Error: SWE-bench harness is required but not available. Please install swebench package."
        return result

    MAX_ENV_UNAVAILABLE_RETRIES = int(os.environ["SWE_MAX_ENV_RETRIES"])
    ENV_UNAVAILABLE_RETRY_WAIT = int(os.environ["SWE_ENV_RETRY_WAIT"])

    resource_level = 1
    for env_attempt in range(1, MAX_ENV_UNAVAILABLE_RETRIES + 1):
        attempt_result = dict(result)  # Fresh copy for each attempt
        try:
            return await _run_tests_with_swebench_harness(
                instance=instance,
                patch=patch,
                config=config,
                result=attempt_result,
                fail_to_pass=fail_to_pass,
                pass_to_pass=pass_to_pass,
                resource_level=resource_level,
            )
        except EnvironmentUnavailable as e:
            logger.error(
                f"[SWE_REWARD] EnvironmentUnavailable during reward evaluation "
                f"(attempt {env_attempt}/{MAX_ENV_UNAVAILABLE_RETRIES}): {e}"
            )
            resource_level += 1
            if env_attempt < MAX_ENV_UNAVAILABLE_RETRIES:
                logger.info(
                    f"[SWE_REWARD] Retrying reward evaluation in {ENV_UNAVAILABLE_RETRY_WAIT}s..."
                )
                await asyncio.sleep(ENV_UNAVAILABLE_RETRY_WAIT)
            else:
                logger.error(
                    f"[SWE_REWARD] All {MAX_ENV_UNAVAILABLE_RETRIES} retry attempts exhausted"
                )
                attempt_result["error"] = (
                    f"Environment unavailable after {MAX_ENV_UNAVAILABLE_RETRIES} attempts: {e}"
                )
                return attempt_result
        except Exception as e:
            logger.error(
                f"[SWE_REWARD] Unexpected error during reward evaluation for "
                f"instance_id={instance.get('instance_id', 'unknown')}: {e}\n{traceback.format_exc()}"
            )
            if env_attempt < MAX_ENV_UNAVAILABLE_RETRIES:
                logger.info(
                    f"[SWE_REWARD] Retrying reward evaluation in {ENV_UNAVAILABLE_RETRY_WAIT}s..."
                )
                await asyncio.sleep(ENV_UNAVAILABLE_RETRY_WAIT)
            else:
                logger.error(
                    f"[SWE_REWARD] All {MAX_ENV_UNAVAILABLE_RETRIES} retry attempts exhausted"
                )
                attempt_result["error"] = f"Unexpected error: {e}"
                return attempt_result

from swebench.harness.test_spec.create_scripts import (
    make_eval_script_list,
)

from .azure_modal_docker import AzureDockerEnvironment
from .swe_wrapper_v2 import EnvironmentUnavailable, _is_environment_error
from .swe_prm import _parse_patch_files, parse_trajectory_from_sample

logger = logging.getLogger(__name__)


async def _execute(env, command: str) -> dict:
    """Execute a command with timeout handling.

    Args:
        env: The execution environment (AzureDockerEnvironment or sync docker env).
        command: The shell command to execute.

    Returns:
        Dict with at least 'output' and 'returncode' keys.

    Raises:
        asyncio.TimeoutError: If the command exceeds the timeout.
    """

    try:
        if isinstance(env, AzureDockerEnvironment):
            result = await asyncio.wait_for(env.execute(command), timeout=SWE_TIMEOUT_REWARD_EXECUTE)
        else:
            result = await asyncio.wait_for(
                asyncio.to_thread(env.execute, command), timeout=SWE_TIMEOUT_REWARD_EXECUTE
            )
        # Detect irrecoverable environment errors (e.g., container deleted/expired)
        output_text = result.get("output", "")
        if result.get("returncode") != 0 and _is_environment_error(output_text):
            raise EnvironmentUnavailable(output_text)
        return result
    except asyncio.TimeoutError:
        cmd_preview = command[:200] + ("..." if len(command) > 200 else "")
        logger.error(
            "[SWE_REWARD] _execute TIMED OUT after %ds for command: %s",
            SWE_TIMEOUT_REWARD_EXECUTE,
            cmd_preview,
        )
        return {"output": f"Command timed out after {SWE_TIMEOUT_REWARD_EXECUTE}s", "returncode": -1, "timed_out": True}


# Maximum base64 payload size per shell command. The agent exec endpoint passes
# the entire command string as a single argv to bash, so this must stay well
# below the sandbox's ARG_MAX. Some sandboxes have very small effective limits
# (~128KB total argv+env), so we keep chunks conservative.
_B64_CHUNK_LIMIT = 32_000

async def _write_file_to_sandbox(env, path: str, content: str | bytes) -> dict:
    """Write content into the sandbox using chunked base64 writes.

    Always uses base64 + heredoc chunking so we can transport arbitrary bytes
    (patches with special characters, embedded delimiters, etc.) without ever
    exceeding the sandbox's per-command argv size limit. The chunk size is
    deliberately small to stay safely under ARG_MAX in restricted sandboxes.

    Returns the result dict from the final _execute call.
    """
    if isinstance(content, str):
        raw = content.encode()
    else:
        raw = content
    encoded = base64.b64encode(raw).decode()

    quoted_path = shlex.quote(path)
    quoted_b64 = shlex.quote(path + ".b64")

    # Clear any stale temp files from a previous run.
    await _execute(env, f"rm -f {quoted_path} {quoted_b64}")

    # Append base64 chunks via heredoc. Chunks are base64-only, so they cannot
    # contain the delimiter B64EOF.
    last_result: dict = {}
    for i in range(0, len(encoded), _B64_CHUNK_LIMIT):
        chunk = encoded[i:i + _B64_CHUNK_LIMIT]
        last_result = await _execute(
            env,
            f"cat >> {quoted_b64} << 'B64EOF'\n{chunk}\nB64EOF",
        )
        if last_result.get("returncode", 0) != 0:
            return last_result

    return await _execute(
        env,
        f"base64 -d {quoted_b64} > {quoted_path} && rm -f {quoted_b64}",
    )


def make_test_spec_scaleswe(
    instance: SWEbenchInstance,
    namespace: Optional[str] = None,
    base_image_tag: str = LATEST,
    env_image_tag: str = LATEST,
    instance_image_tag: str = LATEST,
) -> TestSpec:
    """Create a TestSpec for Scale-SWE instances that don't use MAP_REPO_VERSION_TO_SPECS.

    Scale-SWE instances use f2p_patch/f2p_script instead of the standard test_patch,
    and provide pre-built Docker images rather than repo version specs.
    """
    if isinstance(instance, TestSpec):
        return instance
    assert base_image_tag is not None, "base_image_tag cannot be None"
    assert env_image_tag is not None, "env_image_tag cannot be None"
    assert instance_image_tag is not None, "instance_image_tag cannot be None"
    instance_id = instance[KEY_INSTANCE_ID]
    repo = instance.get("repo", "")
    version = instance.get("version")

    def _from_json_or_obj(key: str) -> Any:
        """If key points to string, load with json"""
        if key not in instance:
            # If P2P, F2P keys not found, it's a validation instance
            return []
        if isinstance(instance[key], str):
            return json.loads(instance[key])
        return instance[key]

    pass_to_pass = _from_json_or_obj("PASS_TO_PASS")
    fail_to_pass = _from_json_or_obj("FAIL_TO_PASS")

    # For Scale-SWE, use f2p_patch (introduces failing test expectations);
    # fall back to test_patch for standard SWE-bench instances.
    test_patch = instance.get("f2p_patch") or instance.get("test_patch", "")

    repo_directory = instance.get("workdir", "/testbed")

    # Build test command with explicit F2P + P2P test IDs (same as reference evaluator)
    base_test_cmd = "pytest --no-header -rA --tb=line --color=no -p no:cacheprovider -W ignore::DeprecationWarning"
    all_test_ids = fail_to_pass + pass_to_pass
    if all_test_ids:
        test_id_str = " ".join(shlex.quote(tid) for tid in all_test_ids)
        test_command = f"{base_test_cmd} {test_id_str}"
    else:
        test_command = base_test_cmd

    install_config = {
        "env_vars": None,
        "env_yml_path": None,
        "install": "",
        "log_parser": "parse_log_pytest_scaleswe",
        "no_use_env": None,
        "packages": "",
        "pip_packages": [],
        "pre_install": None,
        "python": "3.11",
        "reqs_path": [],
        "test_cmd": test_command,
    }

    docker_specs = {}
    repo_script_list = []
    env_script_list = []
    HEREDOC_DELIMITER = "EOF_114329324912"

    eval_script_list = [
        f"cd {repo_directory}",
    ]
    # Only apply the test patch (f2p_patch) if non-empty
    if test_patch:
        eval_script_list.append(
            f"cat > /tmp/f2p_patch.diff <<'{HEREDOC_DELIMITER}'\n{test_patch}\n{HEREDOC_DELIMITER}"
        )
        eval_script_list.append(
            f"cd {repo_directory} && "
            f"(git apply --verbose /tmp/f2p_patch.diff || "
            f"git apply --verbose --reject /tmp/f2p_patch.diff || "
            f"patch --batch --forward --fuzz=5 -p1 -i /tmp/f2p_patch.diff || "
            f"true)"
        )
    eval_script_list.extend([
        f": '{START_TEST_OUTPUT}'",
        test_command,
        f": '{END_TEST_OUTPUT}'",
    ])

    arch = "x86_64"

    image_name = instance.get("image_url") or instance.get("image_name")

    return TestSpec(
        instance_id=instance_id,
        repo=repo,
        env_script_list=env_script_list,
        repo_script_list=repo_script_list,
        eval_script_list=eval_script_list,
        version=version,
        arch=arch,
        FAIL_TO_PASS=fail_to_pass,
        PASS_TO_PASS=pass_to_pass,
        language='py',
        docker_specs=docker_specs,
        namespace=namespace,
        base_image_tag=base_image_tag,
        env_image_tag=env_image_tag,
        instance_image_tag=instance_image_tag,
        install_config=install_config,
        image_name=image_name,
    )


def make_test_spec_swerebench_v2(
    instance: SWEbenchInstance,
    namespace: Optional[str] = None,
    base_image_tag: str = LATEST,
    env_image_tag: str = LATEST,
    instance_image_tag: str = LATEST,
) -> TestSpec:
    """Create a TestSpec for SWE-rebench-V2 instances.

    Rebench-V2 instances provide pre-built Docker images via ``image_url``,
    a structured ``install_config`` with ``test_cmd`` and ``log_parser``,
    and a standard ``test_patch`` for introducing failing-test expectations.
    """
    if isinstance(instance, TestSpec):
        return instance
    assert base_image_tag is not None, "base_image_tag cannot be None"
    assert env_image_tag is not None, "env_image_tag cannot be None"
    assert instance_image_tag is not None, "instance_image_tag cannot be None"

    instance_id = instance[KEY_INSTANCE_ID]
    repo = instance.get("repo", "")
    version = instance.get("version")

    def _from_json_or_obj(key: str) -> Any:
        if key not in instance:
            return []
        if isinstance(instance[key], str):
            return json.loads(instance[key])
        return instance[key]

    pass_to_pass = _from_json_or_obj("PASS_TO_PASS")
    fail_to_pass = _from_json_or_obj("FAIL_TO_PASS")

    test_patch = instance.get("test_patch", "")
    repo_directory = instance.get("workdir", "/testbed")

    # Use install_config from instance (contains test_cmd, log_parser, etc.)
    inst_install_config = instance.get("install_config", {})
    test_command = inst_install_config.get("test_cmd")
    log_parser = inst_install_config.get("log_parser", "parse_log_pytest")

    install_config = {
        "env_vars": None,
        "env_yml_path": None,
        "install": inst_install_config.get("install", ""),
        "log_parser": log_parser,
        "no_use_env": None,
        "packages": "",
        "pip_packages": [],
        "pre_install": None,
        "python": "3.10",
        "reqs_path": [],
        "test_cmd": test_command,
    }

    docker_specs = inst_install_config.get("docker_specs", {})
    repo_script_list = [install_config.get("install", "")]
    env_script_list = []
    HEREDOC_DELIMITER = "EOF_114329324912"

    eval_script_list = [
        f"cd {repo_directory}",
    ]
    # Apply test_patch if non-empty
    if test_patch:
        eval_script_list.append(
            f"cat > /tmp/test_patch.diff <<'{HEREDOC_DELIMITER}'\n{test_patch}\n{HEREDOC_DELIMITER}"
        )
        eval_script_list.append(
            f"cd {repo_directory} && "
            f"(git apply --verbose /tmp/test_patch.diff || "
            f"git apply --verbose --reject /tmp/test_patch.diff || "
            f"patch --batch --forward --fuzz=5 -p1 -i /tmp/test_patch.diff || "
            f"true)"
        )
    eval_script_list.extend([
        f": '{START_TEST_OUTPUT}'",
        test_command,
        f": '{END_TEST_OUTPUT}'",
    ])

    arch = "x86_64"

    # Resolve image name: rebench-v2 uses "image_url" field
    image_name = instance.get("image_url") or instance.get("image_name")

    return TestSpec(
        instance_id=instance_id,
        repo=repo,
        env_script_list=env_script_list,
        repo_script_list=repo_script_list,
        eval_script_list=eval_script_list,
        version=version,
        arch=arch,
        FAIL_TO_PASS=fail_to_pass,
        PASS_TO_PASS=pass_to_pass,
        language="py",
        docker_specs=docker_specs,
        namespace=namespace,
        base_image_tag=base_image_tag,
        env_image_tag=env_image_tag,
        instance_image_tag=instance_image_tag,
        install_config=install_config,
        image_name=image_name,
    )


async def _run_tests_with_swebench_harness(
    instance: dict,
    patch: str,
    config: dict,
    result: dict,
    fail_to_pass: list[str] | None = None,
    pass_to_pass: list[str] | None = None,
    resource_level: int = 1,
) -> dict:
    """
    Async implementation of SWE-bench harness evaluation.

    This follows the same approach as run_instance_modal():
    1. Create TestSpec from instance
    2. Apply patch with git apply (fallback to patch command)
    3. Run test_spec.eval_script
    4. Grade with get_eval_report()
    """
    import time
    reward_start_time = time.time()
    instance_id = instance.get("instance_id", "unknown")
    logger.info(f"[SWE_REWARD] Starting _run_tests_with_swebench_harness for instance_id={instance_id}")

    # Create TestSpec from instance (this contains eval_script and all test info)
    if instance.get("dataset", "") == "AweAI-Team/Scale-SWE":
        test_spec = make_test_spec_scaleswe(instance)
    elif instance.get("dataset", "") == "nebius/SWE-rebench-V2":
        test_spec = make_test_spec_swerebench_v2(instance)
    else:
        try:
            test_spec = make_test_spec(instance)
        except Exception as e:
            result["error"] = f"Failed to create TestSpec: {e}"
            logger.error(f"[SWE_REWARD] Failed to create TestSpec for instance_id={instance_id}: {e}")
            return result


    # Get environment config
    env_config = config.get("environment", {}).copy()
    env_config["environment_class"] = env_config.get("environment_class", "docker")

    # Increase execution timeout for reward evaluation (original exe_timeout + 300s)
    env_config["exe_timeout"] = env_config.get("exe_timeout", 60) + 300

    # Scale CPU and memory based on resource level (only increased for EnvironmentUnavailable)
    env_config["cpu"] = str(resource_level * 2)
    env_config["memory"] = f"{resource_level * 8}Gi"
    logger.info(f"[SWE_REWARD] Environment resources for resource_level {resource_level}: cpu={env_config['cpu']}, memory={env_config['memory']}")

    # Resolve workdir: use instance's workdir if available, otherwise default to /testbed
    workdir = instance.get("workdir", "/testbed")
    env_config.setdefault("cwd", workdir)
    logger.info(f"[SWE_REWARD] Using workdir={workdir} for instance_id={instance_id}")

    # Build startup command: append pre_commands from instance if available
    startup_command = config.get("run", {}).get("env_startup_command")

    # Timeout for creating the reward evaluation environment
    MAX_CREATE_ENV_RETRIES = 1
    RETRY_WAIT_SECONDS = 5
    env = None

    # Create Docker environment with correct SWE-bench image (with retry logic)
    logger.info(f"[SWE_REWARD] Creating test environment for instance_id={instance_id}")
    _env_create_start = time.time()
    for attempt in range(1, MAX_CREATE_ENV_RETRIES + 1):
        logger.info(f"[SWE_REWARD] Attempt {attempt}/{MAX_CREATE_ENV_RETRIES} to create environment for instance_id={instance_id}")
        try:
            env = await asyncio.wait_for(
                create_environment(
                    env_config,
                    instance_id=f"test-{instance_id}",
                    instance=instance,
                    startup_command=startup_command,
                ),
                timeout=SWE_TIMEOUT_REWARD_CREATE_ENV,
            )
            break  # Success, exit retry loop
        except asyncio.TimeoutError:
            logger.error(
                f"[SWE_REWARD] TIMEOUT creating test environment for instance_id={instance_id} "
                f"after {SWE_TIMEOUT_REWARD_CREATE_ENV}s (attempt {attempt}/{MAX_CREATE_ENV_RETRIES})"
            )
            if attempt < MAX_CREATE_ENV_RETRIES:
                logger.info(
                    f"[SWE_REWARD] Retrying in {RETRY_WAIT_SECONDS}s for instance_id={instance_id}"
                )
                await asyncio.sleep(RETRY_WAIT_SECONDS)
            else:
                logger.error(
                    f"[SWE_REWARD] All {MAX_CREATE_ENV_RETRIES} attempts exhausted for instance_id={instance_id}"
                )
                result["error"] = (
                    f"Timed out creating test environment after {SWE_TIMEOUT_REWARD_CREATE_ENV}s "
                    f"({MAX_CREATE_ENV_RETRIES} attempts)"
                )
                return result
        except Exception as e:
            logger.error(
                f"[SWE_REWARD] Failed to create test environment for instance_id={instance_id}: {e} "
                f"(attempt {attempt}/{MAX_CREATE_ENV_RETRIES})"
            )
            if attempt < MAX_CREATE_ENV_RETRIES:
                logger.info(
                    f"[SWE_REWARD] Retrying in {RETRY_WAIT_SECONDS}s for instance_id={instance_id}"
                )
                await asyncio.sleep(RETRY_WAIT_SECONDS)
            else:
                logger.error(
                    f"[SWE_REWARD] All {MAX_CREATE_ENV_RETRIES} attempts exhausted for instance_id={instance_id}"
                )
                result["error"] = f"Failed to create test environment: {e} ({MAX_CREATE_ENV_RETRIES} attempts)"
                return result
    assert env is not None, "Environment creation failed without raising an exception"
    
    logger.info(f"[SWE_REWARD] Test environment created for instance_id={instance_id}, elapsed={time.time() - _env_create_start:.2f}s")

    test_output_path = None
    try:
        # Write patch to a file in the container
        # Using base64 + chunking to handle arbitrarily large patches and any special chars
        logger.info(f"[SWE_REWARD] [{instance_id}] Writing patch to container...")
        # # Previous version used a heredoc to write the patch directly into the container
        # write_patch_cmd = f"cat > /tmp/patch.diff << 'SWEAGENT_PATCH_EOF'\n{patch}\nSWEAGENT_PATCH_EOF"
        # await _execute(env, write_patch_cmd)
        await _write_file_to_sandbox(env, "/tmp/patch.diff", patch)
        logger.info(f"[SWE_REWARD] [{instance_id}] Patch written, applying with git apply...")

        # Apply the patch using multiple fallback strategies (following scaleswe evaluator)
        patch_applied = False
        for git_apply_cmd in [
            f"cd {workdir} && git apply --verbose /tmp/patch.diff",
            f"cd {workdir} && git apply --verbose --reject /tmp/patch.diff",
            f"cd {workdir} && patch --batch --fuzz=5 -p1 -i /tmp/patch.diff",
        ]:
            apply_result = await _execute(env, git_apply_cmd)
            if apply_result.get("returncode", 0) == 0:
                patch_applied = True
                break
            logger.warning("[SWE_REWARD] [%s] Patch apply failed with: %s", instance_id, git_apply_cmd)

        if not patch_applied:
            result["error"] = f"Failed to apply patch: {apply_result.get('output', '')}"
            return result

        # Get git diff before running eval script (for debugging)
        _git_diff_before = await _execute(env, f"cd {workdir} && git -c core.fileMode=false diff")

        f2p_script = instance.get("f2p_script", "")
        if f2p_script:
            write_f2p_cmd = f"cat > {workdir}/test_fail_to_pass.py << 'SWEAGENT_F2P_SCRIPT_EOF'\n{f2p_script}\nSWEAGENT_F2P_SCRIPT_EOF"
            await _execute(env, write_f2p_cmd)

        # Write and run the eval script from TestSpec
        eval_script = test_spec.eval_script
        # Django hack (same as run_instance_modal)
        eval_script = eval_script.replace("locale-gen", "locale-gen en_US.UTF-8")

        logger.debug("Eval script for %s:\n%s", instance_id, eval_script)

        write_eval_cmd = f"cat > /root/eval.sh << 'EVAL_EOF'\n{eval_script}\nEVAL_EOF"
        await _execute(env, write_eval_cmd)
        await _execute(env, "chmod +x /root/eval.sh")

        run_command = f"cd {workdir}"
        # pylint hack
        if "pylint" in test_spec.instance_id:
            run_command += " && PYTHONPATH="
        # increase recursion limit for testing
        run_command += " && python3 -c 'import sys; sys.setrecursionlimit(10000)'"
        # Redirect stderr to stdout so that set -x traces (which contain the
        # >>>>> Start/End Test Output markers) are interleaved with the actual
        # test output.  AzureDockerEnvironment captures stdout and stderr
        # separately and concatenates them (stdout first), which breaks the
        # marker-based splitting in get_logs_eval when 2>&1 is omitted.
        run_command += " && /bin/bash /root/eval.sh 2>&1"

        # run eval script
        _eval_start = time.time()
        logger.info(f"[SWE_REWARD] [{instance_id}] Running eval script...")

        test_result = await _execute(env, run_command)
        logger.info(f"[SWE_REWARD] [{instance_id}] Eval script finished in {time.time() - _eval_start:.2f}s")
        output = test_result.get("output", "")
        result["output"] = output

        # Get git diff after running eval script (for debugging)
        _git_diff_after = await _execute(env, f"cd {workdir} && git -c core.fileMode=false diff")

        # Write test output to a temporary file for get_eval_report
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(output)
            test_output_path = Path(f.name)

        # Build prediction dict for get_eval_report (same format as run_instance_modal)
        pred = {
            KEY_INSTANCE_ID: instance_id,
            KEY_MODEL: "slime-swe-agent",
            KEY_PREDICTION: patch,
        }

        # Use get_eval_report for proper SWE-bench grading
        report = get_eval_report(
            test_spec=test_spec,
            prediction=pred,
            test_log_path=test_output_path,
            include_tests_status=True,
        )

        # Extract results from report
        instance_report = report.get(instance_id, {})
        resolved = instance_report.get("resolved", False)

        result["passed"] = resolved
        result["resolved"] = resolved
        result["resolution"] = ResolvedStatus.FULL.value if resolved else ResolvedStatus.NO.value

        # Extract detailed test results if available
        tests_status = instance_report.get("tests_status", {})
        f2p_status = tests_status.get(FAIL_TO_PASS, {})
        p2p_status = tests_status.get(PASS_TO_PASS, {})

        f2p_passed = len(f2p_status.get("success", []))
        f2p_failed = len(f2p_status.get("failure", []))
        p2p_passed = len(p2p_status.get("success", []))
        p2p_failed = len(p2p_status.get("failure", []))

        result["f2p_total"] = f2p_passed + f2p_failed
        result["f2p_passed"] = f2p_passed
        result["p2p_total"] = p2p_passed + p2p_failed
        result["p2p_passed"] = p2p_passed
        result["tests_passed"] = f2p_passed + p2p_passed
        result["tests_run"] = result["f2p_total"] + result["p2p_total"]

        # Calculate rates
        if result["f2p_total"] > 0:
            result["fail_to_pass_rate"] = f2p_passed / result["f2p_total"]
        else:
            result["fail_to_pass_rate"] = 1.0  # No F2P tests means all "passing"

        if result["p2p_total"] > 0:
            result["pass_to_pass_rate"] = p2p_passed / result["p2p_total"]
        else:
            result["pass_to_pass_rate"] = 1.0  # No P2P tests means all "passing"

    except EnvironmentUnavailable:
        raise  # Propagate to retry loop in run_tests_in_docker()
    except Exception as e:
        result["error"] = f"Test execution error: {str(e)}\n{traceback.format_exc()}"
        logger.error(f"[SWE_REWARD] Test execution error for instance_id={instance_id}: {e}")
    finally:
        # Always cleanup the environment
        if env is not None:
            logger.info(f"[SWE_REWARD] Stopping test environment for instance_id={instance_id}")
            try:
                await asyncio.wait_for(stop_environment(env), timeout=SWE_TIMEOUT_REWARD_EXECUTE)
                logger.info(f"[SWE_REWARD] Test environment stopped for instance_id={instance_id}")
            except asyncio.TimeoutError:
                logger.error(f"[SWE_REWARD] TIMEOUT stopping test environment for instance_id={instance_id}")
            except Exception as e:
                logger.error(f"[SWE_REWARD] Failed to stop test environment for instance_id={instance_id}: {e}")
        else:
            logger.warning(f"[SWE_REWARD] No environment to stop for instance_id={instance_id} (env was None)")
        # Cleanup temporary file
        if test_output_path is not None and test_output_path.exists():
            try:
                test_output_path.unlink()
            except Exception:
                pass

    result["reward_eval_time"] = time.time() - reward_start_time
    logger.info(
        "[SWE_REWARD] Test results for %s: resolved=%s f2p=%.3f p2p=%.3f tests=%s error=%s total_time=%.2fs",
        instance_id,
        result.get("resolved", False),
        result.get("fail_to_pass_rate", 0.0),
        result.get("pass_to_pass_rate", 0.0),
        result.get("tests_run", 0),
        bool(result.get("error")),
        result["reward_eval_time"],
    )
    return result


# ==============================================================================
# Process-shaping helpers (CODE-BASED, deterministic — section A + integrity gate G)
# ==============================================================================
#
# These power the ``process_shaped`` reward mode.  Design principle: outcome
# (resolved / F2P rate) stays primary; process terms only *densify* zero-variance
# GRPO groups and *anchor* shaping to ground truth.  Only signals validated to
# predict success are included (e.g. localization AUC≈0.58).  Anything that
# regressed to AUC≈0.5 in offline analysis (generic verify-before-submit,
# generic reproduce, raw patch-length penalty) is deliberately excluded.
#
# All helpers are pure / cheap (no Docker, no network): they read the already
# computed ``test_results`` dict and the parsed trajectory.  The trajectory is
# parsed once via ``swe_prm.parse_trajectory_from_sample`` (lazy import to avoid
# a hard dependency cycle and to keep import cost off the hot path).

# Heuristic: file paths that look like test / fixture files (used by A5 and G).
_TEST_FILE_RE = re.compile(
    r"(^|/)(tests?|testing)(/|$)|(^|/)conftest\.py$|(^|/)test_[^/]*\.py$|[^/]*_test\.py$",
    re.IGNORECASE,
)

# Destructive shell commands that should trip the integrity gate G.
_DESTRUCTIVE_CMD_RE = re.compile(
    r"\brm\s+-rf\b|\bgit\s+reset\s+--hard\b|\bgit\s+checkout\s+--\s|\bgit\s+clean\s+-[a-z]*f"
    r"|\b(chmod|chown)\s+-R\b|>\s*/dev/sd|\bmkfs\b|\bdd\s+if=",
    re.IGNORECASE,
)

# Reward-hacking signatures in the *submitted patch* that should trip gate G:
# hardcoding an expected value / special-casing a specific failing input rather
# than fixing the root cause.  These are intentionally conservative (high
# precision, low recall) — the offline LLM judge (section B) catches the rest.
_HARDCODE_PATCH_RE = re.compile(
    r"^\+.*\b(pytest\.skip|pytest\.xfail|@pytest\.mark\.(skip|xfail))\b"
    r"|^\+\s*assert\s+True\b"
    r"|^\+\s*return\s+(True|None)\s*#",
    re.IGNORECASE | re.MULTILINE,
)


def _is_test_path(path: str) -> bool:
    """Return True if a repo-relative path looks like a test / fixture file."""
    return bool(_TEST_FILE_RE.search(path or ""))


def _localization_score(patch_files: list[str], gold_files: list[str]) -> float:
    """A2 (part 1): fraction of gold source files the submission actually touched.

    ``|submitted ∩ gold| / |gold|`` restricted to NON-test gold files (editing
    test files is graded by the integrity gate, not rewarded here).  Returns 0.0
    when there is no gold signal so the term is simply inert on datasets without
    a gold patch.
    """
    gold_src = [f for f in gold_files if not _is_test_path(f)]
    if not gold_src:
        return 0.0
    submitted = set(patch_files)
    hit = sum(1 for f in gold_src if f in submitted)
    return hit / len(gold_src)


def _read_before_edit_bonus(traj, gold_files: list[str]) -> float:
    """A2 (part 2): did the agent *read* a gold source file before its first edit?

    This is the genuine *process* half of the localization signal (grounding),
    as opposed to the outcome-correlated patch∩gold half.  Returns 1.0 if any
    gold source file appears in a read-style command that occurs strictly before
    the first edit-style command; else 0.0.  No gold signal -> 0.0 (inert).
    """
    gold_src = [f for f in gold_files if not _is_test_path(f)]
    if not gold_src or traj is None:
        return 0.0

    read_re = re.compile(r"\b(cat|less|head|tail|sed -n|grep|rg|nl|vi|view|open|read_file)\b")
    edit_re = re.compile(r"\b(sed -i|>|>>|tee|patch|apply_patch|git apply|python - <<|cat >)\b|edit_file")

    first_edit_idx = None
    for s in traj.steps:
        cmd = s.command or ""
        if edit_re.search(cmd):
            first_edit_idx = s.idx
            break

    for s in traj.steps:
        if first_edit_idx is not None and s.idx >= first_edit_idx:
            break
        cmd = s.command or ""
        if read_re.search(cmd) and any(g in cmd for g in gold_src):
            return 1.0
    return 0.0


def _extra_files_penalty(patch_files: list[str], gold_files: list[str]) -> float:
    """A5: fraction of submitted files that are outside the gold set (over-reach).

    Counts NON-test submitted files not present in the gold patch, normalized by
    the number of submitted source files.  This is a cleaner "over-modification"
    signal than raw patch length (length penalizes genuinely hard fixes).  When
    there is no gold signal, returns 0.0 (inert).
    """
    if not gold_files:
        return 0.0
    gold_set = set(gold_files)
    submitted_src = [f for f in patch_files if not _is_test_path(f)]
    if not submitted_src:
        return 0.0
    extra = sum(1 for f in submitted_src if f not in gold_set)
    return extra / len(submitted_src)


# Pytest result line patterns, e.g. ``tests/test_x.py::test_foo PASSED`` or
# ``FAILED tests/test_x.py::test_foo``.  Captures the nodeid and the status.
_PYTEST_LINE_RE = re.compile(
    r"(?:^|\s)(?P<node1>[\w./\-]+::[\w\[\]\-./ ]+?)\s+(?P<stat1>PASSED|FAILED|ERROR)\b"
    r"|(?:^|\s)(?P<stat2>PASSED|FAILED|ERROR)\s+(?P<node2>[\w./\-]+::[\w\[\]\-./ ]+)",
    re.MULTILINE,
)


def _test_states_in_text(text: str) -> dict[str, str]:
    """Extract a {test_nodeid: status} map from a chunk of test output."""
    states: dict[str, str] = {}
    if not text:
        return states
    for m in _PYTEST_LINE_RE.finditer(text):
        node = (m.group("node1") or m.group("node2") or "").strip()
        stat = (m.group("stat1") or m.group("stat2") or "").strip().upper()
        if node:
            states[node] = stat
    return states


def _detect_flip(traj, gold_f2p: list[str] | None = None) -> float:
    """A3: anchored reproduce→fix loop.

    Rewards a *flip* — a test observed FAILING earlier in the trajectory and
    PASSING later — NOT merely "ran a test" (that signal had AUC≈0.52 and is
    excluded).  Prefers tests whose nodeid matches a gold FAIL_TO_PASS name so
    the verification is anchored to the true target; falls back to any test that
    flips (agent-built self-check).  Returns 1.0 if such a flip exists else 0.0.
    """
    if traj is None:
        return 0.0
    gold_f2p = gold_f2p or []
    # Normalize gold F2P nodeids to their bare test-name tail for fuzzy matching.
    gold_names = set()
    for g in gold_f2p:
        gold_names.add(g)
        gold_names.add(g.split("::")[-1].split("[")[0])

    seen_fail: set[str] = set()
    any_flip = False
    for s in traj.steps:
        states = _test_states_in_text(s.observation or "")
        for node, stat in states.items():
            tail = node.split("::")[-1].split("[")[0]
            anchored = node in gold_names or tail in gold_names
            if stat in ("FAILED", "ERROR"):
                seen_fail.add(node)
            elif stat == "PASSED" and node in seen_fail:
                # A previously-failing test now passes -> flip.
                if anchored or not gold_names:
                    return 1.0
                any_flip = True
    # If we have gold names but only un-anchored flips, give partial credit.
    return 1.0 if any_flip and not gold_names else (0.5 if any_flip else 0.0)


def _error_rate(traj) -> float:
    """A7: fraction of executed steps whose command returned a non-zero code.

    Pure operational-hygiene loss signal (current baseline ≈14%).  Steps with an
    unknown returncode (None) are ignored from the denominator.
    """
    if traj is None:
        return 0.0
    rcs = [s.returncode for s in traj.steps if s.returncode is not None]
    if not rcs:
        return 0.0
    bad = sum(1 for rc in rcs if rc != 0)
    return bad / len(rcs)


def _no_progress_penalty(traj) -> float:
    """A6: penalize *patterns* of no-progress churn (NOT step count).

    Penalizes, in [0,1]:
      - exact repeated commands (same command string run again), and
      - repeated reads of the same file with no intervening edit.
    Step count itself is deliberately NOT penalized (it is inversely correlated
    with success — hard tasks need more steps).
    """
    if traj is None or not traj.steps:
        return 0.0
    cmds = [(s.command or "").strip() for s in traj.steps]
    nonempty = [c for c in cmds if c]
    if not nonempty:
        return 0.0
    seen: set[str] = set()
    repeats = 0
    for c in nonempty:
        if c in seen:
            repeats += 1
        else:
            seen.add(c)
    return min(1.0, repeats / len(nonempty))


def _integrity_gate(patch: str, traj, gold_test_files: list[str] | None = None) -> float:
    """Integrity gate G: multiplicative cap in {1.0 (clean), 0.0 (violation)}.

    Trips (returns 0.0) when the submission or trajectory shows reward-hacking /
    destructive behavior:
      - editing test / conftest / fixture files (incl. known gold test files),
      - hardcoded expected values / skip / xfail / ``assert True`` in the patch,
      - destructive shell commands (``rm -rf``, ``git reset --hard``, ...).
    This is a CAP, not a bonus: clean trajectories get 1.0 and are unaffected.
    """
    gold_test_files = set(gold_test_files or [])

    # 1) Edited test files (path heuristic OR known gold test file).
    for f in _parse_patch_files(patch):
        if f in gold_test_files or _is_test_path(f):
            return 0.0

    # 2) Reward-hacking signatures inside the patch body.
    if patch and _HARDCODE_PATCH_RE.search(patch):
        return 0.0

    # 3) Destructive commands anywhere in the trajectory.
    if traj is not None:
        for s in traj.steps:
            if s.command and _DESTRUCTIVE_CMD_RE.search(s.command):
                return 0.0

    return 1.0


# ==============================================================================
# Reward Computation Functions
# ==============================================================================

async def compute_simple_reward(args, sample: Sample) -> float:
    """
    Compute simple binary reward based on SWE-bench resolution.

    Returns 1.0 if FULL resolution (all F2P tests pass and all P2P tests maintained),
    -1.0 otherwise.

    Args:
        args: Training arguments
        sample: Sample with response and metadata

    Returns:
        Binary reward (-1.0 or 1.0)
    """
    # Extract final output from metadata
    final_output = sample.metadata.get("final_output", "")

    if not final_output:
        return -1.0

    # Extract patch
    patch = extract_patch_from_output(final_output)

    if not is_valid_patch(patch):
        return -1.0

    # Get test information from metadata
    instance_id = sample.metadata.get("instance_id", "unknown")
    repo = sample.metadata.get("repo", "")
    base_commit = sample.metadata.get("base_commit", "")

    # Get SWE-bench test lists
    fail_to_pass = sample.metadata.get("FAIL_TO_PASS", [])
    pass_to_pass = sample.metadata.get("PASS_TO_PASS", [])

    # Handle JSON string format
    if isinstance(fail_to_pass, str):
        try:
            fail_to_pass = json.loads(fail_to_pass)
        except json.JSONDecodeError:
            fail_to_pass = []
    if isinstance(pass_to_pass, str):
        try:
            pass_to_pass = json.loads(pass_to_pass)
        except json.JSONDecodeError:
            pass_to_pass = []

    # Run tests with full instance data for proper Docker image resolution
    test_results = await run_tests_in_docker(
        instance_id=instance_id,
        patch=patch,
        repo=repo,
        base_commit=base_commit,
        fail_to_pass=fail_to_pass,
        pass_to_pass=pass_to_pass,
        instance=sample.metadata,
    )

    # final_reward = 1.0 if test_results["resolved"] else 0.0
    # final_reward *= 10.0

    # Assign a positive reward for resolved tests and a configurable negative reward
    # for unresolved tests (controlled via SWE_UNRESOLVED_REWARD env var, default -1.0)
    unresolved_reward = float(os.environ.get("SWE_UNRESOLVED_REWARD", "-1.0"))
    final_reward = 1.0 if test_results["resolved"] else unresolved_reward

    # update test results with final reward for debugging
    test_results["final_reward"] = final_reward
    # write test results to trajectory path for debugging
    trajectory_path = sample.metadata.get("trajectory_path", "unknown")
    if trajectory_path != "unknown":
        try:
            with open(trajectory_path.replace(".json", "_rewards.json"), "w") as f:
                json.dump(test_results, f)
                logger.info(f"test_results: {json.dumps(test_results)}")
        except Exception as e:
            logger.error(f"Failed to write test results to trajectory path {trajectory_path}: {e}")
    
    return final_reward


# Only "Submitted" indicates the agent finished the trajectory on its own.
# Any other exit_status (LimitsExceeded, TimeExceeded, Timeout_LLM, Error_LLM,
# Unknown, ...) is treated as a truncated / incomplete trajectory.
SUBMITTED_EXIT_STATUS = "Submitted"


async def compute_simple_truncated_reward(args, sample: Sample) -> float:
    """
    Compute binary reward like ``compute_simple_reward`` but penalize truncated
    trajectories more strongly.

    Reward scheme:
    - -2.0 if the trajectory did not finish with exit_status == "Submitted"
      (covers LimitsExceeded, TimeExceeded, LLM errors, unknown, etc.)
    -  1.0 if the patch fully resolves the SWE-bench instance
    - -1.0 otherwise
    """
    exit_status = (
        sample.metadata.get("exit_status", "Unknown") if sample.metadata else "Unknown"
    )

    # Anything other than a clean "Submitted" is treated as truncated.
    if exit_status != SUBMITTED_EXIT_STATUS:
        instance_id = sample.metadata.get("instance_id", "unknown") if sample.metadata else "unknown"
        logger.info(
            f"[SWE_REWARD] Trajectory not submitted (exit_status={exit_status}) "
            f"for instance_id={instance_id}, assigning reward=-2.0"
        )
        truncated_results = {
            "truncated": True,
            "exit_status": exit_status,
            "final_reward": -2.0,
        }
        trajectory_path = sample.metadata.get("trajectory_path", "unknown") if sample.metadata else "unknown"
        if trajectory_path != "unknown":
            try:
                with open(trajectory_path.replace(".json", "_rewards.json"), "w") as f:
                    json.dump(truncated_results, f)
                    logger.info(f"test_results: {json.dumps(truncated_results)}")
            except Exception as e:
                logger.error(f"Failed to write test results to trajectory path {trajectory_path}: {e}")
        return -2.0

    # Otherwise fall back to the standard simple binary reward (+1 / -1).
    return await compute_simple_reward(args, sample)


async def compute_simple_truncated_zero_reward(args, sample: Sample) -> float:
    """
    Variant of ``compute_simple_truncated_reward`` that uses 0.0 (instead of
    -1.0) as the default penalty for unresolved-but-submitted trajectories.

    Reward scheme:
    - -2.0 if the trajectory did not finish with exit_status == "Submitted"
      (covers LimitsExceeded, TimeExceeded, LLM errors, unknown, etc.)
    -  1.0 if the patch fully resolves the SWE-bench instance
    -  0.0 otherwise
    """
    exit_status = (
        sample.metadata.get("exit_status", "Unknown") if sample.metadata else "Unknown"
    )

    # Anything other than a clean "Submitted" is treated as truncated.
    if exit_status != SUBMITTED_EXIT_STATUS:
        instance_id = sample.metadata.get("instance_id", "unknown") if sample.metadata else "unknown"
        logger.info(
            f"[SWE_REWARD] Trajectory not submitted (exit_status={exit_status}) "
            f"for instance_id={instance_id}, assigning reward=-2.0"
        )
        truncated_results = {
            "truncated": True,
            "exit_status": exit_status,
            "final_reward": -2.0,
        }
        trajectory_path = sample.metadata.get("trajectory_path", "unknown") if sample.metadata else "unknown"
        if trajectory_path != "unknown":
            try:
                with open(trajectory_path.replace(".json", "_rewards.json"), "w") as f:
                    json.dump(truncated_results, f)
                    logger.info(f"test_results: {json.dumps(truncated_results)}")
            except Exception as e:
                logger.error(f"Failed to write test results to trajectory path {trajectory_path}: {e}")
        return -2.0

    # Submitted: run the simple binary evaluation, then map -1.0 -> 0.0.
    simple_reward = await compute_simple_reward(args, sample)
    return 1.0 if simple_reward > 0 else 0.0


async def compute_simple_truncated_minus_one_reward(args, sample: Sample) -> float:
    """
    Variant of ``compute_simple_truncated_reward`` with a milder truncation
    penalty (-1.0 instead of -2.0).

    Reward scheme:
    - -1.0 if the trajectory did not finish with exit_status == "Submitted"
      (covers LimitsExceeded, TimeExceeded, LLM errors, unknown, etc.)
    -  1.0 if the patch fully resolves the SWE-bench instance
    -  SWE_UNRESOLVED_REWARD (env, default -1.0) for submitted-but-unresolved
    """
    exit_status = (
        sample.metadata.get("exit_status", "Unknown") if sample.metadata else "Unknown"
    )

    # Anything other than a clean "Submitted" is treated as truncated.
    if exit_status != SUBMITTED_EXIT_STATUS:
        instance_id = sample.metadata.get("instance_id", "unknown") if sample.metadata else "unknown"
        logger.info(
            f"[SWE_REWARD] Trajectory not submitted (exit_status={exit_status}) "
            f"for instance_id={instance_id}, assigning reward=-1.0"
        )
        truncated_results = {
            "truncated": True,
            "exit_status": exit_status,
            "final_reward": -1.0,
        }
        trajectory_path = sample.metadata.get("trajectory_path", "unknown") if sample.metadata else "unknown"
        if trajectory_path != "unknown":
            try:
                with open(trajectory_path.replace(".json", "_rewards.json"), "w") as f:
                    json.dump(truncated_results, f)
                    logger.info(f"test_results: {json.dumps(truncated_results)}")
            except Exception as e:
                logger.error(f"Failed to write test results to trajectory path {trajectory_path}: {e}")
        return -1.0

    # Submitted: resolved -> 1.0, unresolved -> SWE_UNRESOLVED_REWARD.
    return await compute_simple_reward(args, sample)


async def compute_simple_plus_llm_as_judges_reward(args, sample: Sample) -> float:
    """Binary outcome reward PLUS an LLM-as-judge
    process reward applied to ALL paths.

    Reward scheme (with PRM_ALPHA = alpha, R_unres = SWE_UNRESOLVED_REWARD):
      - truncated (R_bin = -1.0):                 R' = -1.0    + alpha * J
      - submitted, unresolved (R_bin = R_unres):  R' = R_unres + alpha * J
      - submitted, resolved   (R_bin =  1.0):     R' =  1.0    + alpha * J

    where J in [0, 1] is the aggregated judge score from
    ``swe_prm.score_sample``. Outcome ordering (resolved > unresolved/truncated)
    is preserved as long as ``alpha`` is small enough that the resolved region
    [1.0, 1.0 + alpha] does not overlap the others. Note that with the default
    ``SWE_UNRESOLVED_REWARD = -1.0`` the truncated and submitted-unresolved
    regions coincide.

    Short-circuits (no API call) when ``PRM_ALPHA == 0``: falls back to pure
    ``compute_simple_truncated_minus_one_reward``.

    On any PRM failure, falls back to the underlying binary reward unchanged.
    Configuration is read from ``PRM_*`` env vars (see ``swe_prm.py``).
    """
    instance_id = (
        sample.metadata.get("instance_id", "unknown") if sample.metadata else "unknown"
    )

    base_reward = await compute_simple_truncated_minus_one_reward(args, sample)

    # alpha == 0 -> short-circuit: behave exactly like the underlying binary
    # reward, skipping the (slow, costly) judge API call entirely.
    try:
        alpha = float(os.environ.get("PRM_ALPHA", "0.5"))
    except ValueError:
        alpha = 0.5
    if alpha == 0.0:
        logger.info(
            f"[PRM] {instance_id}: PRM_ALPHA=0 -> skipping judge, R={base_reward:.3f}"
        )
        if sample.metadata is not None:
            sample.metadata["prm"] = {"called": False, "alpha": 0.0}
        return base_reward

    process_reward = 0.0
    error_str = ""
    scores: dict = {}
    try:
        from .swe_prm import score_sample

        prm_timeout = float(os.environ.get("PRM_TIMEOUT_TOTAL", "120"))
        grade = await asyncio.wait_for(
            score_sample(sample, timeout=max(prm_timeout * 0.8, 10.0)),
            timeout=prm_timeout,
        )
        process_reward = float(grade.process_reward)
        scores = grade.scores or {}
        error_str = grade.error or ""
        if error_str:
            logger.warning(
                f"[PRM] {instance_id}: judge returned error: {error_str}"
            )
    except asyncio.TimeoutError:
        error_str = "timeout"
        logger.warning(
            f"[PRM] {instance_id}: TIMEOUT after {os.environ.get('PRM_TIMEOUT_TOTAL', '120')}s; J=0.0"
        )
    except Exception as e:
        error_str = f"{type(e).__name__}: {e}"
        logger.warning(f"[PRM] {instance_id}: failed: {error_str}; J=0.0")

    final_reward = base_reward + alpha * process_reward
    logger.info(
        f"[PRM] {instance_id}: R_binary={base_reward:.3f}, J={process_reward:.3f}, "
        f"alpha={alpha}, R_final={final_reward:.3f}"
    )

    # Write structured PRM info onto sample.metadata so that
    # _compute_prm_metrics (rollout.py) can aggregate it into wandb.
    if sample.metadata is not None:
        sample.metadata["prm"] = {
            "called": True,
            "alpha": alpha,
            "base_binary_reward": base_reward,
            "process_reward": process_reward,
            "final_reward": final_reward,
            "contribution": alpha * process_reward,
            "scores": scores,           # {"R1" fix quality, "R2".."R7" self-verification}
            "error": error_str,         # "" == success, "timeout" == timed-out, else exception
        }

    # Augment the *_rewards.json that compute_simple_reward / the truncated
    # wrapper already wrote, so PRM info is auditable per trajectory.
    trajectory_path = (
        sample.metadata.get("trajectory_path", "unknown")
        if sample.metadata
        else "unknown"
    )
    if trajectory_path != "unknown":
        rewards_json_path = trajectory_path.replace(".json", "_rewards.json")
        try:
            existing: dict = {}
            if os.path.exists(rewards_json_path):
                with open(rewards_json_path) as f:
                    existing = json.load(f)
            existing.update(
                {
                    "base_binary_reward": base_reward,
                    "process_reward": process_reward,
                    "prm_alpha": alpha,
                    "final_reward": final_reward,
                    "judge_details": {
                        "scores": scores,
                        "error": error_str,
                    },
                }
            )
            with open(rewards_json_path, "w") as f:
                json.dump(existing, f)
        except Exception as e:
            logger.error(
                f"[PRM] Failed to update rewards json {rewards_json_path}: {e}"
            )

    return final_reward


async def compute_pass_rate_reward(args, sample: Sample) -> float:
    """
    Compute reward based on test pass rate.

    This provides a more granular reward signal based on:
    - Fail-to-Pass rate (weighted higher as it's the primary objective)
    - Pass-to-Pass rate (maintenance)

    Formula: 0.7 * F2P_rate + 0.3 * P2P_rate

    Args:
        args: Training arguments
        sample: Sample with response and metadata

    Returns:
        Pass rate reward (0.0 to 1.0)
    """
    final_output = sample.metadata.get("final_output", "")
    #final_output = sample.label # !!!!!!!!!!!!!!!! only for debugging

    if not final_output:
        return 0.0

    patch = extract_patch_from_output(final_output)

    if not is_valid_patch(patch):
        return 0.0

    instance_id = sample.metadata.get("instance_id", "unknown")
    repo = sample.metadata.get("repo", "")
    base_commit = sample.metadata.get("base_commit", "")

    fail_to_pass = sample.metadata.get("FAIL_TO_PASS", [])
    pass_to_pass = sample.metadata.get("PASS_TO_PASS", [])

    if isinstance(fail_to_pass, str):
        try:
            fail_to_pass = json.loads(fail_to_pass)
        except json.JSONDecodeError:
            fail_to_pass = []
    if isinstance(pass_to_pass, str):
        try:
            pass_to_pass = json.loads(pass_to_pass)
        except json.JSONDecodeError:
            pass_to_pass = []

    test_results = await run_tests_in_docker(
        instance_id=instance_id,
        patch=patch,
        repo=repo,
        base_commit=base_commit,
        fail_to_pass=fail_to_pass,
        pass_to_pass=pass_to_pass,
        instance=sample.metadata,
    )

    if test_results.get("error"):
        return 0.0

    # Weighted combination of F2P and P2P rates
    f2p_weight = getattr(args, "swe_f2p_weight", 0.3)
    p2p_weight = getattr(args, "swe_p2p_weight", 0.7)

    f2p_rate = test_results.get("fail_to_pass_rate", 0.0)
    p2p_rate = test_results.get("pass_to_pass_rate", 0.0)

    final_reward = f2p_weight * f2p_rate + p2p_weight * p2p_rate

    final_reward *= 10.0

    # update test results with final reward for debugging
    test_results["final_reward"] = final_reward
    # write test results to trajectory path for debugging
    trajectory_path = sample.metadata.get("trajectory_path", "unknown")
    if trajectory_path != "unknown":
        try:
            with open(trajectory_path.replace(".json", "_rewards.json"), "w") as f:
                json.dump(test_results, f)
                logger.info(f"test_results: {json.dumps(test_results)}")
        except Exception as e:
            logger.error(f"Failed to write test results to trajectory path {trajectory_path}: {e}")

    return final_reward


async def compute_multi_objective_reward(args, sample: Sample) -> dict[str, float]:
    """
    Compute multi-objective reward with multiple components.

    Components:
    - resolved: Binary reward for FULL resolution
    - fail_to_pass_rate: Rate of F2P test success
    - pass_to_pass_rate: Rate of P2P test success
    - patch_valid: Reward for valid patch syntax
    - step_efficiency: Reward for using fewer steps
    - token_efficiency: Reward for using fewer tokens

    Args:
        args: Training arguments
        sample: Sample with response and metadata

    Returns:
        Dict of reward components
    """
    rewards = {}

    final_output = sample.metadata.get("final_output", "")
    n_steps = sample.metadata.get("n_steps", 0)
    exit_status = sample.metadata.get("exit_status", "Unknown")
    max_steps = getattr(args, "swe_max_steps", 100)

    # Extract patch
    patch = extract_patch_from_output(final_output)
    rewards["patch_valid"] = 1.0 if is_valid_patch(patch) else 0.0

    # Completion reward
    rewards["completed"] = 1.0 if exit_status == "Submitted" else 0.0

    # Step efficiency
    if n_steps > 0:
        rewards["step_efficiency"] = max(0.0, 1.0 - (n_steps / max_steps))
    else:
        rewards["step_efficiency"] = 0.0

    # Token efficiency
    response_length = sample.response_length
    max_response_length = getattr(args, "rollout_max_response_len", 16384)
    if response_length > 0:
        rewards["token_efficiency"] = max(0.0, 1.0 - (response_length / max_response_length))
    else:
        rewards["token_efficiency"] = 0.0

    # Test-based rewards
    if not final_output or not is_valid_patch(patch):
        rewards["resolved"] = 0.0
        rewards["fail_to_pass_rate"] = 0.0
        rewards["pass_to_pass_rate"] = 0.0
    else:
        instance_id = sample.metadata.get("instance_id", "unknown")
        repo = sample.metadata.get("repo", "")
        base_commit = sample.metadata.get("base_commit", "")

        fail_to_pass = sample.metadata.get("FAIL_TO_PASS", [])
        pass_to_pass = sample.metadata.get("PASS_TO_PASS", [])

        if isinstance(fail_to_pass, str):
            try:
                fail_to_pass = json.loads(fail_to_pass)
            except json.JSONDecodeError:
                fail_to_pass = []
        if isinstance(pass_to_pass, str):
            try:
                pass_to_pass = json.loads(pass_to_pass)
            except json.JSONDecodeError:
                pass_to_pass = []

        test_results = await run_tests_in_docker(
            instance_id=instance_id,
            patch=patch,
            repo=repo,
            base_commit=base_commit,
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
            instance=sample.metadata,
        )

        rewards["resolved"] = 1.0 if test_results["resolved"] else 0.0
        rewards["fail_to_pass_rate"] = test_results.get("fail_to_pass_rate", 0.0)
        rewards["pass_to_pass_rate"] = test_results.get("pass_to_pass_rate", 0.0)

        # update test results with final reward for debugging
        test_results["rewards"] = rewards
        # write test results to trajectory path for debugging
        trajectory_path = sample.metadata.get("trajectory_path", "unknown")
        if trajectory_path != "unknown":
            try:
                with open(trajectory_path.replace(".json", "_rewards.json"), "w") as f:
                    json.dump(test_results, f)
                    logger.info(f"test_results: {json.dumps(test_results)}")
            except Exception as e:
                logger.error(f"Failed to write test results to trajectory path {trajectory_path}: {e}")

    return rewards


# ==============================================================================
# Process-shaped reward (mode: "process_shaped")
# ==============================================================================
#
# Composite reward: outcome stays primary, validated process terms densify
# zero-variance GRPO groups and anchor shaping to ground truth, and a
# multiplicative integrity gate caps reward-hacking / destructive behavior.
#
#   R = ( w0·resolved + w1·f2p_rate + w2·localization + w3·anchored_flip
#         + w4·p2p_rate − p5·extra_files − p6·no_progress − p7·error_rate )
#       × integrity_gate            (gate ∈ {1.0, 0.0})
#
# Weight ordering enforced by defaults: w0 ≫ {w1,w2,w3} > {w4} > {p5,p6,p7}.
# All weights are configurable via ``getattr(args, ...)`` then ``PROCESS_*`` env
# vars (mirrors the existing ``swe_f2p_weight`` pattern) so terms can be retuned
# or disabled per run and re-validated by AUC without code edits.

_PROCESS_WEIGHT_DEFAULTS = {
    "resolved": 1.0,          # w0  outcome (primary)
    "f2p_rate": 0.5,          # w1  dense outcome (A1)
    "localization": 0.4,      # w2  validated grounding (A2, AUC≈0.58)
    "anchored_flip": 0.3,     # w3  reproduce→fix loop (A3)
    "read_before_edit": 0.1,  # w2b process half of localization
    "p2p_rate": 0.2,          # w4  non-regression (A4)
    "extra_files": 0.05,      # p5  over-reach penalty (A5)
    "no_progress": 0.05,      # p6  churn-pattern penalty (A6)
    "error_rate": 0.03,       # p7  operational hygiene (A7)
}

# Components that are subtracted (penalties) rather than added.
_PROCESS_PENALTY_KEYS = {"extra_files", "no_progress", "error_rate"}


def _process_weight(args, key: str) -> float:
    """Resolve a process-shaping weight: args attr > env var > default."""
    attr = getattr(args, f"swe_process_{key}_weight", None)
    if attr is not None:
        return float(attr)
    env = os.environ.get(f"PROCESS_{key.upper()}_WEIGHT")
    if env is not None:
        try:
            return float(env)
        except ValueError:
            pass
    return _PROCESS_WEIGHT_DEFAULTS.get(key, 0.0)


def combine_process_rewards(args, rewards: dict[str, float]) -> float:
    """Combine process-shaped components with a multiplicative integrity gate.

    Additive terms minus penalty terms, then multiplied by the gate (so a gate
    of 0.0 zeroes the whole reward), then scaled ×10 for training (matching the
    other reward modes).
    """
    total = 0.0
    for key, default in _PROCESS_WEIGHT_DEFAULTS.items():
        w = _process_weight(args, key)
        val = rewards.get(key, 0.0)
        if key in _PROCESS_PENALTY_KEYS:
            total -= w * val
        else:
            total += w * val

    gate = rewards.get("integrity_gate", 1.0)
    total *= gate
    total *= 10.0
    return total


async def compute_process_shaped_reward(args, sample: Sample) -> dict[str, float]:
    """Compute the validated process-shaped reward components (section A + gate G).

    Returns a dict of components (so it can be logged / AUC-validated per term).
    Use :func:`combine_process_rewards` to reduce it to the training scalar.

    Components:
      - resolved, f2p_rate, p2p_rate          (outcome / dense outcome, A1/A4)
      - localization, read_before_edit        (A2: patch∩gold + grounding)
      - anchored_flip                         (A3: reproduce→fix loop)
      - extra_files, no_progress, error_rate  (A5/A6/A7 penalties)
      - integrity_gate                        (G: multiplicative cap)
    """
    rewards: dict[str, float] = {
        "resolved": 0.0,
        "f2p_rate": 0.0,
        "p2p_rate": 0.0,
        "localization": 0.0,
        "read_before_edit": 0.0,
        "anchored_flip": 0.0,
        "extra_files": 0.0,
        "no_progress": 0.0,
        "error_rate": 0.0,
        "integrity_gate": 1.0,
    }

    final_output = sample.metadata.get("final_output", "")
    patch = extract_patch_from_output(final_output)

    # Parse trajectory once (cheap, no Docker). Tolerate any parsing failure so
    # the reward never crashes a rollout — fall back to outcome-only shaping.
    traj = None
    try:
        traj = parse_trajectory_from_sample(sample)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[SWE_REWARD] process_shaped: trajectory parse failed: {e}")

    gold_files = list(getattr(traj, "gold_changed_files", []) or [])
    gold_test_files = list(getattr(traj, "gold_test_files", []) or [])

    # Gold FAIL_TO_PASS names for anchoring the flip detector (A3).
    fail_to_pass = sample.metadata.get("FAIL_TO_PASS", [])
    if isinstance(fail_to_pass, str):
        try:
            fail_to_pass = json.loads(fail_to_pass)
        except json.JSONDecodeError:
            fail_to_pass = []

    # ---- Trajectory-derived terms (work even when outcome is all-fail) -------
    patch_files = _parse_patch_files(patch)
    rewards["localization"] = _localization_score(patch_files, gold_files)
    rewards["read_before_edit"] = _read_before_edit_bonus(traj, gold_files)
    rewards["anchored_flip"] = _detect_flip(traj, fail_to_pass)
    rewards["extra_files"] = _extra_files_penalty(patch_files, gold_files)
    rewards["no_progress"] = _no_progress_penalty(traj)
    rewards["error_rate"] = _error_rate(traj)
    rewards["integrity_gate"] = _integrity_gate(patch, traj, gold_test_files)

    # ---- Outcome terms (require a valid patch + Docker eval) -----------------
    test_results: dict = {}
    if final_output and is_valid_patch(patch):
        instance_id = sample.metadata.get("instance_id", "unknown")
        repo = sample.metadata.get("repo", "")
        base_commit = sample.metadata.get("base_commit", "")

        pass_to_pass = sample.metadata.get("PASS_TO_PASS", [])
        if isinstance(pass_to_pass, str):
            try:
                pass_to_pass = json.loads(pass_to_pass)
            except json.JSONDecodeError:
                pass_to_pass = []

        test_results = await run_tests_in_docker(
            instance_id=instance_id,
            patch=patch,
            repo=repo,
            base_commit=base_commit,
            fail_to_pass=fail_to_pass if isinstance(fail_to_pass, list) else [],
            pass_to_pass=pass_to_pass if isinstance(pass_to_pass, list) else [],
            instance=sample.metadata,
        )

        rewards["resolved"] = 1.0 if test_results.get("resolved") else 0.0
        rewards["f2p_rate"] = test_results.get("fail_to_pass_rate", 0.0)
        rewards["p2p_rate"] = test_results.get("pass_to_pass_rate", 0.0)

    # ---- Debug dump ----------------------------------------------------------
    test_results = dict(test_results)
    test_results["rewards"] = rewards
    test_results["process_scalar"] = combine_process_rewards(args, rewards)
    trajectory_path = sample.metadata.get("trajectory_path", "unknown")
    if trajectory_path != "unknown":
        try:
            with open(trajectory_path.replace(".json", "_rewards.json"), "w") as f:
                json.dump(test_results, f)
                logger.info(f"test_results: {json.dumps(test_results)}")
        except Exception as e:
            logger.error(f"Failed to write test results to trajectory path {trajectory_path}: {e}")

    return rewards


async def compute_shaped_reward(args, sample: Sample) -> dict[str, float]:
    """
    Compute shaped reward with intermediate progress signals.

    This provides denser reward signal by rewarding:
    - File exploration (ls, find, cat)
    - File editing (vim, sed, patch)
    - Test execution (pytest, python -m test)
    - Valid patch generation
    - SWE-bench pass rates

    Args:
        args: Training arguments
        sample: Sample with response and metadata

    Returns:
        Dict of reward components including shaping signals
    """
    # Start with multi-objective rewards
    rewards = await compute_multi_objective_reward(args, sample)

    # Add shaping rewards based on trajectory
    trajectory = sample.metadata.get("trajectory", {})
    messages = trajectory.get("messages", [])

    # Track intermediate progress
    explored_files = False
    edited_files = False
    ran_tests = False

    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")

            # Check for file exploration
            if any(cmd in content for cmd in ["ls", "find", "cat", "grep"]):
                explored_files = True

            # Check for file editing
            if any(cmd in content for cmd in ["vim", "sed", "patch", "echo >", ">"]):
                edited_files = True

            # Check for test execution
            if any(cmd in content for cmd in ["pytest", "python -m test", "python -m unittest"]):
                ran_tests = True

    # Add intermediate rewards (small bonuses)
    rewards["explored_files"] = 0.1 if explored_files else 0.0
    rewards["edited_files"] = 0.2 if edited_files else 0.0
    rewards["ran_tests"] = 0.1 if ran_tests else 0.0

    return rewards


def combine_rewards(rewards: dict[str, float], weights: dict[str, float] | None = None) -> float:
    """
    Combine multiple reward components into a single scalar.

    Default weights prioritize test success, with small bonuses for other factors.

    Args:
        rewards: Dict of reward components
        weights: Optional custom weights for each component

    Returns:
        Combined scalar reward
    """
    default_weights = {
        "resolved": 1.0,              # Primary: full resolution
        "fail_to_pass_rate": 0.5,     # F2P test pass rate
        "pass_to_pass_rate": 0.3,     # P2P test pass rate
        "patch_valid": 0.1,           # Valid patch bonus
        "completed": 0.1,             # Completion bonus
        "step_efficiency": 0.05,      # Efficiency bonus
        "token_efficiency": 0.05,     # Token efficiency bonus
        "explored_files": 0.02,       # Shaping: exploration
        "edited_files": 0.03,         # Shaping: editing
        "ran_tests": 0.02,            # Shaping: testing
    }

    if weights is None:
        weights = default_weights

    total = 0.0
    for key, value in rewards.items():
        weight = weights.get(key, 0.0)
        total += weight * value

    total *= 10.0  # Scale up for training

    return total


async def reward_func(args, sample: Sample, **kwargs) -> float:
    """
    Main reward function for SWE-bench tasks.

    Supports multiple reward modes:
    - "simple": Binary reward (1.0 for FULL resolution, 0.0 otherwise)
    - "simple_truncated": Same as "simple", but trajectories truncated by
      step/token/time limits receive -2.0 instead of running the evaluator
    - "simple_truncated_zero": Same as "simple_truncated", but unresolved
      (yet submitted) trajectories get 0.0 instead of -1.0
    - "simple_plus_llm_as_judges": "simple_truncated_zero" binary R, plus an
      LLM-as-judge process reward J in [0,1] added to ALL paths:
      R' = R_binary + PRM_ALPHA * J. Configured via PRM_* env vars (see
      ``swe_prm.py``). When PRM_ALPHA == 0 it short-circuits to
      "simple_truncated_zero" (no API call). PRM failure falls back to R_binary.
    - "pass_rate": Weighted pass rate (F2P and P2P rates)
    - "multi_objective": Multiple reward components combined into scalar
    - "shaped": Includes intermediate progress rewards combined into scalar
    - "process_shaped": Outcome (resolved/F2P) primary, plus *validated*
      deterministic process shaping (localization A2, anchored reproduce->fix
      flip A3, P2P non-regression A4) minus small over-reach/churn/error
      penalties (A5/A6/A7), all multiplied by an integrity gate G that caps
      reward-hacking / destructive trajectories. See
      ``compute_process_shaped_reward`` / ``combine_process_rewards``.

    Eval-mode override:
      When called with ``evaluation=True`` (forwarded by slime's eval rollout
      via ``async_rm``), the active mode is replaced with the value of the
      ``SWE_EVAL_REWARD_MODE`` env var (default ``simple_truncated_zero``).
      This lets training use ``simple_plus_llm_as_judges`` while eval tracks
      pure solve-rate without paying for the PRM API.

    Args:
        args: Training arguments (should have swe_reward_mode attribute)
        sample: Sample to compute reward for
        **kwargs: Additional keyword arguments

    Returns:
        Float reward value (always a scalar for compatibility with training framework)
    """
    
    instance_id = sample.metadata.get("instance_id", "unknown") if sample.metadata else "unknown"
    logger.info(f"[SWE_REWARD] Starting reward_func for instance_id={instance_id}")
    
    # Get reward mode (priority: args > environment variable > default)
    reward_mode = getattr(args, "swe_reward_mode", None)
    if reward_mode is None:
        reward_mode = os.environ.get("SWE_REWARD_MODE", "simple")

    # During eval rollouts we want to track pure solve-rate, not the
    # PRM-augmented (mixed) reward used for training.  slime forwards
    # ``evaluation=True`` from sglang_rollout.eval_rollout_single_dataset
    # -> async_rm -> here via **kwargs.  When set, we override the active
    # reward mode with SWE_EVAL_REWARD_MODE (default "simple_truncated_zero"),
    # which is binary and never calls the PRM API.  Set the env var to the
    # same value as SWE_REWARD_MODE if you actually want PRM during eval.
    is_eval = bool(kwargs.get("evaluation", False))
    if is_eval:
        eval_mode = os.environ.get("SWE_EVAL_REWARD_MODE", "simple_truncated_zero")
        if eval_mode != reward_mode:
            logger.info(
                f"[SWE_REWARD] {instance_id}: eval rollout -> overriding "
                f"reward_mode '{reward_mode}' -> '{eval_mode}' (no PRM API call)"
            )
        reward_mode = eval_mode

    # Wrap the entire reward computation with a timeout to prevent indefinite hangs
    # (e.g. Azure Docker container creation or eval script execution stuck)
    try:
        if reward_mode == "simple":
            result = await asyncio.wait_for(compute_simple_reward(args, sample), timeout=SWE_TIMEOUT_REWARD_TOTAL)
        elif reward_mode == "simple_truncated":
            result = await asyncio.wait_for(compute_simple_truncated_reward(args, sample), timeout=SWE_TIMEOUT_REWARD_TOTAL)
        elif reward_mode == "simple_truncated_zero":
            result = await asyncio.wait_for(compute_simple_truncated_zero_reward(args, sample), timeout=SWE_TIMEOUT_REWARD_TOTAL)
        elif reward_mode == "simple_plus_llm_as_judges":
            result = await asyncio.wait_for(compute_simple_plus_llm_as_judges_reward(args, sample), timeout=SWE_TIMEOUT_REWARD_TOTAL)
        elif reward_mode == "pass_rate":
            result = await asyncio.wait_for(compute_pass_rate_reward(args, sample), timeout=SWE_TIMEOUT_REWARD_TOTAL)
        elif reward_mode == "multi_objective":
            rewards = await asyncio.wait_for(compute_multi_objective_reward(args, sample), timeout=SWE_TIMEOUT_REWARD_TOTAL)
            result = combine_rewards(rewards)
        elif reward_mode == "shaped":
            rewards = await asyncio.wait_for(compute_shaped_reward(args, sample), timeout=SWE_TIMEOUT_REWARD_TOTAL)
            result = combine_rewards(rewards)
        elif reward_mode == "process_shaped":
            rewards = await asyncio.wait_for(compute_process_shaped_reward(args, sample), timeout=SWE_TIMEOUT_REWARD_TOTAL)
            result = combine_process_rewards(args, rewards)
        else:
            raise ValueError(f"Unknown reward mode: {reward_mode}")
        logger.info(f"[SWE_REWARD] Completed reward_func for instance_id={instance_id}, reward={result}")
        assert isinstance(result, (float, int)), f"Reward function must return a scalar float or int, got {type(result)}"
        return result
    except asyncio.TimeoutError:
        logger.error(
            f"[SWE_REWARD] reward_func TIMED OUT after {SWE_TIMEOUT_REWARD_TOTAL}s for instance_id={instance_id}. "
        )
        return -1.0
    except Exception as e:
        logger.error(f"[SWE_REWARD] reward_func ERROR for instance_id={instance_id}: {e}\n{traceback.format_exc()}")
        return -1.0


# Synchronous wrapper for backward compatibility
def compute_reward_sync(args, sample: Sample, **kwargs) -> float | dict[str, Any]:
    """
    Synchronous wrapper for reward computation.

    Args:
        args: Training arguments
        sample: Sample to compute reward for
        **kwargs: Additional keyword arguments

    Returns:
        Reward value(s)
    """
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # If already in async context, create task
        return asyncio.create_task(reward_func(args, sample, **kwargs))
    else:
        # Run in event loop
        return loop.run_until_complete(reward_func(args, sample, **kwargs))
