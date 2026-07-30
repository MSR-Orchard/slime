"""
Reward computation for browser agent tasks.

Two reward components:
1. Format reward  – deterministic check that the agent response adheres to
   the policy defined in system_prompt.md (<think>, <tool_call>).
2. LLM-as-a-Judge – async OpenAI-compatible call that evaluates whether the
   task was successfully completed, adapted from the original GPT-4V evaluator.

Usage:
    from reward_browser import reward_func

    score = await reward_func(args, sample)
"""

import asyncio
import json
import logging
import os
import random
import re
import time
from datetime import datetime
from typing import Any
from urllib import response

from openai import AsyncAzureOpenAI, AsyncOpenAI

from examples.orchard_gui.base.utils import ToolParser, resolve_parser_type
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

# ==============================================================================
# Constants
# ==============================================================================

# Load valid tool names from tool_list.json
_TOOL_LIST_PATH = os.path.join(os.path.dirname(__file__), "env", "prompts", "tool_list.json")
with open(_TOOL_LIST_PATH, "r") as _f:
    _TOOLS_INFO = json.load(_f)
    VALID_TOOL_NAMES = {tool["function"]["name"] for tool in _TOOLS_INFO}

# Concurrency control (mirrors swe_reward.py)
_REWARD_CONCURRENCY = int(os.environ.get("BROWSER_REWARD_CONCURRENCY", "8"))
_reward_semaphore = asyncio.Semaphore(_REWARD_CONCURRENCY)

# ==============================================================================
# LLM Judge Prompts (adapted from WebVoyager)
# ==============================================================================

JUDGE_SYSTEM_PROMPT = (
    "As an evaluator, you will be presented with three primary components "
    "to assist you in your role:\n\n"
    "1. Web Task Instruction: This is a clear and specific directive provided "
    "in natural language, detailing the online activity to be carried out.\n\n"
    "2. Result Screenshots: This is a visual representation of the screen "
    "showing the result or intermediate state of performing a web task.\n\n"
    "3. Result Response: This is a textual response obtained after the "
    "execution of the web task.\n\n"
    "-- You DO NOT NEED to interact with web pages or perform actions.\n"
    "-- You SHOULD NOT make assumptions based on information not presented "
    "in the screenshot when comparing it to the instructions.\n"
    "-- Your primary responsibility is to conduct a thorough assessment of "
    "the web task instruction against the outcome depicted in the screenshot "
    "and in the response, evaluating whether the actions taken align with "
    "the given instructions.\n"
    "-- NOTE that the instruction may involve more than one task. Failing to "
    "complete either task should be considered unsuccessful.\n"
    "-- NOTE that the screenshot is authentic, but the response provided by "
    "LLM is generated at the end of web browsing; there may be discrepancies "
    "between the text and the screenshots.\n"
    "-- Note the difference: 1) Result response may contradict the screenshot, "
    "then the content of the screenshot prevails, 2) The content in the Result "
    "response is not mentioned on the screenshot, choose to believe the content.\n\n"
    "You should elaborate on how you arrived at your final evaluation and then "
    "provide a definitive verdict on whether the task has been successfully "
    "accomplished, either as 'SUCCESS' or 'NOT SUCCESS'."
)

JUDGE_SYSTEM_PROMPT_ACTION_HISTORY = (
    "As an evaluator, you will be presented with four primary components "
    "to assist you in your role:\n\n"
    "1. Web Task Instruction: This is a clear and specific directive provided "
    "in natural language, detailing the online activity to be carried out.\n\n"
    "2. Agent Action History: This is a overview of the agent's actions and the "
    "corresponding observation for lat few steps. Use it to understand what the "
    "agent tried to do, but do not treat it as ground truth if it conflicts with "
    "the screenshots.\n\n"
    "3. Result Response: This is a textual response obtained after the "
    "execution of the web task.\n\n"
    "-- You DO NOT NEED to interact with web pages or perform actions.\n"
    "-- You SHOULD use the screenshots as the strongest evidence about the "
    "actual page state.\n"
    "-- You SHOULD use the action history to judge whether the agent followed "
    "the instruction and whether the final response is supported by what "
    "happened on screen.\n"
    "-- If the action history conflicts with screenshots, trust the screenshots.\n"
    "-- NOTE that the instruction may involve more than one task. Failing to "
    "complete either task should be considered unsuccessful.\n"
    "-- NOTE that the final response may contradict the screenshots; in that "
    "case the screenshots prevail. If the final response contains information "
    "not visible in the screenshots, you may still consider it only if it is "
    "consistent with the screenshots and action history.\n\n"
    "You should first explain your reasoning with explicit reference to the "
    "instruction, action history, screenshots, and final response. Then provide "
    "a definitive verdict as either 'SUCCESS' or 'NOT SUCCESS'."
)

JUDGE_USER_PROMPT = (
    "### TASK: {task}\n"
    "### Result Response: {answer}\n"
    "### Last {num} screenshots (from oldest to newest): "
)

JUDGE_USER_PROMPT_ACTION_HISTORY = (
    "### TASK: {task}\n"
    "### Agent Action History:\n{action_history}\n\n"
    "### Result Response: {answer}\n"
    "### Last {num} screenshots (from oldest to newest): "
)


# ==============================================================================
# 1. Format Reward
# ==============================================================================

# Module-level ToolParser for format reward checks.
# Uses regex mode (no tools_info) — same parsing as generate_browser.py L328-333.


def _compute_turn_format_reward(_tool_parser,  response: str) -> float:
    """
    Compute a deterministic format reward based on policy adherence.

    Checks each assistant turn for the required format defined in
    system_prompt.md: ``<think>...</think>`` followed by one or more
    ``<tool_call>...</tool_call>`` blocks.

    Note: for thinking models the input already contains the opening
    ``<think>`` token, so the response text starts inside the think
    block (i.e. no leading ``<think>``).  The expected response format
    is therefore::

        ...reasoning text...
        </think>

        <tool_call>
        {"name": "...", "arguments": {...}}
        </tool_call>

    Scoring rubric (each criterion worth 1/3, total = 1.0):
        - </think> tag present (thinking block closed)     → 1/3
        - <tool_call> tag present                          → 1/3
        - Successfully parsed tool call inside <tool_call> → 1/3

    All three must be met AND appear in the correct order
    (</think> before <tool_call>) for a score of 1.0; otherwise 0.0.

    Args:
        response: The assistant's response text (single turn).

    Returns:
        Float: 1.0 if format is correct, 0.0 otherwise.
    """
    # Strip chat-template delimiters that may still be attached to the response.
    response = response.strip()
    if response.endswith("<|im_end|>"):
        response = response[: -len("<|im_end|>")].rstrip()

    # --- Tag presence checks ---
    has_think_close = "</think>" in response
    parsed = _tool_parser.parse(response)
    has_tool_call =  parsed.success and len(parsed.calls) > 0

    return 1.0 if has_think_close and has_tool_call else 0.0

def compute_format_reward(_tool_parser, response: str, turn_level: bool) -> float:
    if turn_level:
        return _compute_turn_format_reward(_tool_parser, response)
    
    # Extract every assistant turn: content between <|im_start|>assistant\n
    # and <|im_end|>.  This is the standard chat format used in the data files.
    turn_pattern = r"<\|im_start\|>assistant\s*\n(.*?)<\|im_end\|>"
    turns = re.findall(turn_pattern, response, re.DOTALL)
    turns = [t.strip() for t in turns if t.strip()]

    if not turns:
        return 0.0

    scores = [_compute_turn_format_reward(_tool_parser, turn) for turn in turns]
    if all(s == 1.0 for s in scores):
        return 1.0
    return 0.0


# ==============================================================================
# 2. LLM-as-a-Judge Reward
# ==============================================================================

def _extract_final_answer(_tool_parser: ToolParser, response: str) -> str:
    parsed = _tool_parser.parse(response)
    if parsed.success and parsed.calls:
        last_call = parsed.calls[-1]
        if last_call.name != "done":
            return ""
        params = (
            json.loads(last_call.parameters) 
            if isinstance(last_call.parameters, str) 
            else last_call.parameters
        )
        return params.get("response", "").strip()
    return ""

def _clean_think_from_action_history(action_history: str) -> str:
    # Remove all content between <think> and </think> tags
    return re.sub(r"<think>.*?</think>", "", action_history, flags=re.DOTALL)

def _clean_img_placeholders_from_action_history(action_history: str) -> str:
    # Remove all occurrences of the screenshot placeholder
    _SCREENSHOT_PLACEHOLDER = "screenshot:\n<|vision_start|><|image_pad|><|vision_end|>\n"
    return action_history.replace(_SCREENSHOT_PLACEHOLDER, "")


def _normalize_served_base_url(base_url: str) -> str:
    base_url = (base_url or "").strip()
    if not base_url:
        raise ValueError("Judge served mode requires JUDGE_API_BASE or JUDGE_API_HOST/JUDGE_API_PORT.")
    if not re.match(r"^https?://", base_url):
        base_url = f"http://{base_url}"
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return base_url

def _get_openai_client(method="token") -> Any:
    """
    Create an AsyncAzureOpenAI client from environment variables.
    Supports both API key and Azure AD token authentication.
    """
    if method == "token":
        resource_name = os.environ.get("AZURE_RESOURCE_NAME")
        api_version = os.environ.get("AZURE_API_VERSION", "2025-04-01-preview")
        topen_path = os.environ.get("AZURE_TOKEN_PATH", "/tmp/aoai_token")
        with open(topen_path) as f:
            token = f.read().strip()
        return AsyncAzureOpenAI(
            azure_ad_token=token,
            api_version=api_version,
            azure_endpoint=f"https://{resource_name}.openai.azure.com/",
        )
    if method == "api_key":
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_API_BASE")
        api_version = os.environ.get("OPENAI_API_VERSION", "2025-04-01-preview")

        kwargs: dict[str, Any] = {"api_key": api_key, "base_url": base_url, "api_version": api_version}
        return AsyncAzureOpenAI(**kwargs)

    if method == "served":
        base_url = os.environ.get("JUDGE_API_BASE")
        if not base_url:
            host = os.environ.get("JUDGE_API_HOST", "").strip()
            port = os.environ.get("JUDGE_API_PORT", "").strip()
            if host and port:
                base_url = f"{host}:{port}"
            elif host:
                base_url = host

        base_url = _normalize_served_base_url(base_url)
        api_key = os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        return AsyncOpenAI(api_key=api_key, base_url=base_url)

    raise ValueError(f"Unsupported judge_api_mode: {method}")


async def compute_judge_reward(
    args: Any,
    sample: Sample,
    _tool_parser: ToolParser = None,
) -> tuple[float | None, str, bool, str | None, bool]:
    """
    Compute LLM-as-a-Judge reward using an OpenAI-compatible API.

    When ``args.judge_prompt_variant`` is ``"action_history"``, includes the
    agent's action history and uses the action-history-aware system prompt.
    Otherwise uses the default prompt with screenshots only.

    Args:
        args: Training arguments (must contain judge config attributes).
        sample: Sample with response and metadata.
        _tool_parser: ToolParser instance (required for action_history variant).

    Returns:
        (score, judge_text, saw_timeout, error_type, api_called) — score is
        1.0 for SUCCESS, 0.0 for NOT SUCCESS, ``None`` if the judge call/parse
        failed (i.e. the result is uninformative about the agent: API
        exhaustion, timeout, refusal, or missing SUCCESS/NOT SUCCESS verdict).
    """
    task = sample.metadata["intent"]
    use_action_history = getattr(args, "judge_prompt_variant", "default") == "action_history"

    # --- Extract answer ---
    answer = _extract_final_answer(_tool_parser, sample.response or "")
    if not answer:
        logger.warning("No answer extracted from response; judge score = 0.0")
        return 0.0, "No answer extracted.", False, None, False

    # --- Collect screenshots ---
    n_judge_screenshots = getattr(args, "judge_max_attached_imgs")
    screenshots = sample.metadata.get("full_image_list", [])[-n_judge_screenshots:]

    # --- Build messages ---
    if use_action_history:
        action_history = _clean_img_placeholders_from_action_history(sample.prompt + sample.response)
        action_history = _clean_think_from_action_history(action_history)
        user_text = JUDGE_USER_PROMPT_ACTION_HISTORY.format(
            task=task, action_history=action_history,
            answer=answer, num=len(screenshots),
        )
        system_prompt = JUDGE_SYSTEM_PROMPT_ACTION_HISTORY
    else:
        user_text = JUDGE_USER_PROMPT.format(
            task=task, answer=answer, num=str(len(screenshots)),
        )
        system_prompt = JUDGE_SYSTEM_PROMPT

    user_content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    for img_b64 in screenshots:
        url = img_b64 if img_b64.startswith("data:") else f"data:image/png;base64,{img_b64}"
        user_content.append({"type": "image_url", "image_url": {"url": url}})

    if getattr(args, "log_judge_output", False):
        logger.info("Judge user prompt: %s", user_text)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    # --- Call LLM judge with retries ---
    api_model = getattr(args, "judge_api_model")
    judge_timeout_secs = getattr(args, "judge_timeout_secs", 120.0)
    judge_api_mode = getattr(args, "judge_api_mode", "token")
    try:
        client = _get_openai_client(method=judge_api_mode)
    except Exception as e:
        logger.warning("Judge API client initialization failed: %s", e)
        return (
            None,
            f"Judge API client initialization failed: {type(e).__name__}.",
            False,
            "client_init_error",
            False,
        )

    max_retries = 3
    saw_timeout = False
    api_called = False
    for attempt in range(max_retries):
        try:
            judge_coro = client.chat.completions.create(
                model=api_model, messages=messages, seed=42,
            )
            api_called = True
            if judge_timeout_secs is not None and judge_timeout_secs > 0:
                response = await asyncio.wait_for(judge_coro, timeout=judge_timeout_secs)
            else:
                response = await judge_coro
            judge_text = response.choices[0].message.content or ""
            logger.info("Judge response: %s", judge_text)

            if "NOT SUCCESS" in judge_text:
                return 0.0, judge_text, False, None, api_called
            if "SUCCESS" in judge_text:
                return 1.0, judge_text, False, None, api_called
            logger.warning("Judge response does not contain SUCCESS/NOT SUCCESS: %s", judge_text)
            return None, judge_text, False, "missing_verdict", api_called

        except asyncio.TimeoutError as e:
            saw_timeout = True
            logger.warning(
                "Judge API call timed out (attempt %d/%d, timeout=%ss): %s",
                attempt + 1, max_retries, judge_timeout_secs, e
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(3 ** (attempt + 1))
            else:
                logger.warning("All judge API retry attempts exhausted due to timeout.")
                return None, f"Judge timeout exhausted after {max_retries} attempts.", True, "timeout", api_called
        except Exception as e:
            logger.warning(
                "Judge API call failed (attempt %d/%d, timeout=%ss): %s",
                attempt + 1, max_retries, judge_timeout_secs, e
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(3 ** (attempt + 1))
            else:
                logger.warning("All judge API retry attempts exhausted.")
                return None, "All judge API retry attempts exhausted.", saw_timeout, "api_error", api_called

    return None, "Unexpected error occurred.", saw_timeout, "api_error", api_called


# ==============================================================================
# 3. Main Reward Function
# ==============================================================================

async def reward_func(args: Any, samples: Sample | list[Sample], **kwargs) -> float | None | list[float | None]:

    if isinstance(samples, Sample):
        return await _score_single_sample(args, samples)

    # For turn-level samples (from generate_turn_sample), compute the reward
    # only on the last turn and share it across all turns in the trajectory.
    is_turn_level = any("turn_index" in s.metadata for s in samples)
    if not is_turn_level:
        raise ValueError("Expected a list of turn-level samples with 'turn_index' in metadata, but none found.")

    # Find the last turn sample
    last_turn_sample = None
    for s in samples:
        if s.metadata.get("is_last_turn", False):
            last_turn_sample = s
            break
    if last_turn_sample is None:
        raise ValueError("No turn sample with is_last_turn=True found in turn-level samples.")

    # Compute reward once on the last turn
    trajectory_reward = await _score_single_sample(args, last_turn_sample)
    # Propagate the same reward and reward metadata to all turns
    reward_meta = last_turn_sample.metadata.get("reward", {})
    for s in samples:
        s.metadata["reward"] = reward_meta
        s.reward = trajectory_reward
        if last_turn_sample.remove_sample:
            s.remove_sample = True
    return [trajectory_reward] * len(samples)


async def _score_single_sample(args: Any, sample: Sample) -> float | None:
    """Score a single sample (called in parallel by reward_func).

    Returns ``None`` when the LLM-judge call/parse failed; the sample is
    also marked ``remove_sample=True`` so the trainer drops it.
    """
    if not isinstance(sample, Sample):
        raise TypeError(f"Expected Sample, got {type(sample)}")

    parser_type = resolve_parser_type(args.hf_checkpoint)
    _tool_parser = ToolParser(_TOOLS_INFO, parser_type=parser_type)

    async with _reward_semaphore:
        # # Determine status from metadata (normalise to lowercase string)
        # raw_status = sample.metadata.get("status", "completed")
        # if isinstance(raw_status, str):
        #     # Handle "Status.ABORTED" → "aborted" format from data files
        #     status = raw_status.rsplit(".", 1)[-1].lower()
        # else:
        #     status = str(raw_status).lower()

        is_turn_level = True if "turn_index" in sample.metadata else False

        # --- Format reward (deterministic) ---
        format_score = compute_format_reward(_tool_parser, sample.response, turn_level=is_turn_level)
    
        # --- LLM-as-a-Judge reward (async) ---
        judge_timeout = False
        judge_required = sample.status == Sample.Status.COMPLETED
        judge_api_called = False
        judge_error_type = None
        if sample.status != Sample.Status.COMPLETED:
            judge_score = 0.0
            judge_text = f"Judge not run for status={sample.status}"
        else:
            (
                judge_score,
                judge_text,
                judge_timeout,
                judge_error_type,
                judge_api_called,
            ) = await compute_judge_reward(args, sample, _tool_parser)
            if judge_score is None:
                # Judge/infra failure (timeout, exhausted retries, refusal, missing verdict):
                # the result is uninformative about the agent, so drop the sample from training.
                sample.remove_sample = True
                if judge_timeout:
                    sample.metadata["judge_timeout"] = True

        # --- Combine (OpenWebRL's -1 / 0 / 1 scheme) ---
        terminate_reason = str((sample.metadata or {}).get("terminate_reason", "")).lower()
        if judge_score is None:
            # Judge/infra failure: the sample is already flagged remove_sample=True above so it won't be trained on.
            # Use 0.0 so slime's GRPO normalization and metrics (which do round()/torch.tensor on rewards) stay crash-safe.
            combined = 0.0
        elif terminate_reason == "consecutive_parse_failures":
            # Trajectory aborted after repeated malformed tool calls (OpenWebRL: terminate_reason
            # "format_error_failed"). Penalize producing unparseable output more than an honest task
            # failure, so GRPO pushes the policy away from garbage output. Tunable via
            # `format_failure_penalty` (default -1.0, OpenWebRL's value; set 0.0 to disable).
            combined = float(getattr(args, "format_failure_penalty", -1.0))
        else:
            combined = 1.0 if (format_score == 1.0 and judge_score == 1.0) else 0.0

        logger.info(
            "Browser reward: format=%.3f judge=%s combined=%s (status=%s)",
            format_score, judge_score, combined, sample.status
        )

        sample.metadata["reward"] = {
            "format": format_score,
            "judge": judge_score,
            "combined": combined,
            "judge_text": judge_text,
            "judge_timeout": judge_timeout,
            "judge_required": judge_required,
            "judge_api_called": judge_api_called,
            "judge_error_type": judge_error_type,
            "judge_prompt_variant": getattr(args, "judge_prompt_variant", "default"),
        }
        sample.reward = combined
        return combined
