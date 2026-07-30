"""
Extensible Gym Environment Integration for slime Training

This module provides a standardized interface for training agents in various
gym-like environments using the slime framework. It uses an adapter pattern
to support multiple environments through a common interface.

Adapted from slime_official/examples/geo3k_vlm_multi_turn/rollout.py
"""

import asyncio
import base64
import io
import logging
import os
import sys
import json
from datetime import datetime
from enum import Enum
from typing import Any

import torch

from PIL import Image

# Suppress noisy asyncio socket warnings that flood logs on connection failures.
# Set level to CRITICAL and install a filter as a safety net in case Ray or
# another library resets the level after import.
_asyncio_logger = logging.getLogger("asyncio")
_asyncio_logger.setLevel(logging.CRITICAL)
_asyncio_logger.addFilter(lambda record: "socket.send()" not in record.getMessage())

# Use fully-qualified imports so that importlib.import_module() in Ray workers
# can resolve them without sys.path hacks.
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO")

from examples.orchard_gui.base.adapter import BaseGymEnvAdapter
from examples.orchard_gui.base.registry import DEFAULT_REGISTRY as ENV_REGISTRY
from examples.orchard_gui.base.utils import ToolParser, resolve_parser_type
from examples.orchard_gui.adapters.browser_adapter import BrowserEnvConfig, BrowserAdapter
from examples.orchard_gui.env.factory import create_env, load_env_config

from slime.utils.types import Sample
from slime.utils.http_utils import post
from slime.rollout.sglang_rollout import GenerateState


async def _initialize_resources(args: Any, task_data: dict):
    """Initialize adapter, generate state, and server URL.

    Args:
        args: Rollout arguments.
        task_data: Task metadata dict (from sample.metadata).
    """
    env_name = "browser"

    # Register browser environment
    env_config = load_env_config(getattr(args, "env_mode", "") or None)

    ENV_REGISTRY.register(
        name=env_name,
        adapter_class=BrowserAdapter,
        config_class=BrowserEnvConfig,
        default_config=env_config,
    )

    env = await create_env(task_data, env_config)

    # Create adapter for this environment
    adapter: BaseGymEnvAdapter = ENV_REGISTRY.create_adapter(name=env_name,
                                                            rollout_args=args,
                                                            config_overrides=None
                                                            )
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    return env, adapter, state, url


async def _run_inference_step(url: str, input_ids: list[int], sampling_params: dict, image_data: list, tokenizer):
    """
    Run a single inference step via SGLang /generate endpoint.
    """
    payload = {
        "input_ids": input_ids,
        "sampling_params": sampling_params,
        "return_logprob": True,
    }
    if image_data:
        payload["image_data"] = image_data

    output = await post(url, payload)
    response_text = output["text"]

    meta = output.get("meta_info", {})
    if "output_token_logprobs" in meta:
        new_tokens = [item[1] for item in meta["output_token_logprobs"]]
        new_log_probs = [item[0] for item in meta["output_token_logprobs"]]
    else:
        raise ValueError("❗  output_token_logprobs not found in meta_info")
        new_tokens, new_log_probs = [], []

    finish_type = meta.get("finish_reason", {}).get("type", "stop")

    response_text = _ensure_im_end_w_new_line(response_text, new_tokens, new_log_probs, tokenizer)

    return response_text, new_tokens, new_log_probs, finish_type


def _ensure_im_end_w_new_line(response_text: str, new_tokens: list[int], new_logprobs: list[float], tokenizer) -> str:
    """
    Ensure the response text and tokens end with "<|im_end|>\n".
    """
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is None or im_end_id == tokenizer.unk_token_id:
        raise ValueError("❗  Tokenizer has no '<|im_end|>' token; cannot normalise turn end.")
    newline_ids = tokenizer.encode("\n", add_special_tokens=False)

    # Normalise text: strip trailing whitespace, append <|im_end|>\n
    response_text = response_text.rstrip()
    if not response_text.endswith("<|im_end|>"):
        response_text += "<|im_end|>"
    response_text += "\n"

    # Normalise tokens: ensure they end with exactly [<|im_end|>, \n] (idempotent —
    # inspect the actual tail, not just the last token, so an existing "<|im_end|>\n" is not duplicated).
    target_tail = [im_end_id, *newline_ids]
    if new_tokens[-len(target_tail):] == target_tail:
        appended = []                       # already ends with <|im_end|>\n
    elif new_tokens[-1] == im_end_id:
        appended = list(newline_ids)        # ends with <|im_end|>, add newline
    else:
        appended = list(target_tail)        # add <|im_end|> + newline
    new_tokens.extend(appended)
    new_logprobs.extend([0.0] * len(appended))

    return response_text

def _ensure_correct_sample(sample: Sample, tokenizer, save_suffix: bool | None=None, debug: bool=False) -> None:
    """
    This function is for debugging: validates that ``sample.tokens`` has the correct total length and optionally
    writes a per-token debug file.

    The check accounts for the fact that ``tokenizer.encode()`` treats each``<|image_pad|>`` placeholder as a single token,
    whereas ``sample.tokens`` stores the fully-expanded visual tokens (``t * h * w // 4`` tokens per image).

    Args:
        sample: Sample whose token stream should be verified.
            - sample.prompt: empty string for trajectory samples; the text fed to the LLM for turn-level samples.
            - sample.response: the LLM's response text for turn-level samples; the complete templated text for trajectory samples.
            - sample.tokens: token IDs with each ``<|image_pad|>`` expanded to the actual number of visual tokens based on image dimensions.
            - sample.metadata["image_grid_thw"]: list of [temporal, height, width] for each image in the sample.
        tokenizer: Tokenizer used to re-encode prompt + response for comparison.
        save_suffix: If not None, writes a tab-separated per-token debug file ``debug_sample{save_suffix}.txt`` (one line per token:
            decoded \\t loss_mask \\t logprob \\t token_id) via _save_per_token_dump.
        debug: If True, also validates the multimodal token-expansion count.

    Raises:
        ValueError: If the length of ``sample.tokens`` does not equal len(tokenizer.encode(prompt + response)) + sum(t*h*w//4) - num_images.
    """
    # loss_mask and rollout_log_probs cover only the response portion (not prompt tokens).
    # Validate they match each other in length.
    if len(sample.loss_mask) != len(sample.rollout_log_probs):
        raise ValueError(
            f"❗  Length mismatch: loss_mask({len(sample.loss_mask)}) != "
            f"rollout_log_probs({len(sample.rollout_log_probs)}). They must be equal."
        )
    if sample.response_length != len(sample.loss_mask):
        raise ValueError(
            f"❗  response_length({sample.response_length}) != "
            f"len(loss_mask)({len(sample.loss_mask)}). They must be equal."
        )
    if debug:
        # compute the total number of image tokens after expansion
        image_grid_thw = sample.multimodal_train_inputs["image_grid_thw"].tolist() if sample.multimodal_train_inputs and "image_grid_thw" in sample.multimodal_train_inputs else []
        cnt_img_tokens = []
        for thw in image_grid_thw:
            cnt_img_tokens.append(thw[0] * thw[1] * thw[2] // 4)

        text_tokens = tokenizer.encode(sample.prompt + sample.response)
        num_images = len(image_grid_thw)
        expected = len(text_tokens) + sum(cnt_img_tokens) - num_images
        actual = len(sample.tokens)
        if actual != expected:
            print(f"{sample.response[-10:]!r}")
            img_tk = tokenizer.encode("<|image_pad|>", add_special_tokens=False)[0]
            text_token_str = "-".join([str(t) for t in text_tokens])
            actual_token_str = "-".join([str(t) for t in sample.tokens])
            # convert continuous multiple image tokens back to one image token
            for cnt_img_tk in cnt_img_tokens:
                actual_token_str = actual_token_str.replace("-".join([str(img_tk)] * cnt_img_tk), str(img_tk))
            print(f"Text tokens ({len(text_tokens)}): {text_token_str}")
            print(f"Actual tokens ({len(sample.tokens)}): {actual_token_str}")

            raise ValueError(
                f"❗  Token length mismatch: sample.tokens has {actual} tokens, "
                f"expected {expected} "
                f"(text_tokens={text_tokens}, cnt_img_tokens={sum(cnt_img_tokens)}, num_images={num_images})"
            )
    
    if save_suffix is not None:
        _save_per_token_dump(sample, tokenizer, f"debug_sample{save_suffix}.txt")


def _append_to_sample(
    sample: Sample,
    new_tokens: list[int],
    logprobs: list[float],
    loss_mask_val: int,
) -> None:
    """Append tokens to sample tracking."""
    if len(new_tokens) != len(logprobs):
        raise ValueError("❗  new_tokens and logprobs must have the same length")
    sample.tokens.extend(new_tokens)
    sample.loss_mask.extend([loss_mask_val] * len(new_tokens))
    if sample.rollout_log_probs is not None:
        sample.rollout_log_probs.extend(logprobs)


def _encode_with_processor(
    processor, tokenizer, text: str, image_data: list[str]
) -> tuple:
    """
    Encode text using the VL processor to correctly expand <|image_pad|> tokens.

    When text contains <|vision_start|><|image_pad|><|vision_end|> placeholders,
    ``tokenizer.encode()`` treats <|image_pad|> as a single token. The processor
    instead expands it to the correct number of tokens based on image dimensions.

    Falls back to ``tokenizer.encode()`` when no processor or no images.

    Args:
        processor: VL processor (e.g. Qwen2VLProcessor) or None
        tokenizer: tokenizer[] instance
        text: formatted text containing image placeholders
        image_data: list of base64 data-URL strings
                    (e.g. "data:image/png;base64,...")

    Returns:
        token_ids: Token IDs with image pad tokens correctly expanded.
        pil_images: list of decoded PIL images (empty list if no images).
        multimodal_train_inputs: processor output minus input_ids/attention_mask,
            suitable for accumulation into sample.multimodal_train_inputs; None
            if no images. Contains image_grid_thw as a tensor.
    """
    if not image_data:
        return tokenizer.encode(text, add_special_tokens=False), [], None
    
    if processor is None:
        raise ValueError("❗  Processor is required for image encoding.")

    # Decode base64 data-URLs → PIL images
    pil_images = []
    for data_url in image_data:
        # Strip the data-URL prefix ("data:image/...;base64,")
        if "," in data_url:
            b64_str = data_url.split(",", 1)[1]
        else:
            b64_str = data_url
        raw_bytes = base64.b64decode(b64_str)
        pil_images.append(Image.open(io.BytesIO(raw_bytes)).convert("RGB"))

    # return_tensors="pt" so pixel_values / image_grid_thw are torch tensors:
    # slime's actor moves each multimodal_train_inputs value to GPU via v.to(device)
    # (slime/backends/megatron_utils/actor.py), which fails on plain Python lists.
    # return_mm_token_type_ids=False: slime concatenates every multimodal_train_inputs
    # key along dim 0; mm_token_type_ids is [1, seq_len] and would fail to cat across
    # samples of different lengths. Matches slime's build_processor_kwargs default.
    processor_output = processor(
        text=[text], images=pil_images, return_tensors="pt", return_mm_token_type_ids=False
    )
    # input_ids is now a tensor; sample.tokens / SGLang need a list[int].
    token_ids = processor_output["input_ids"][0].tolist()

    # Collect processor tensors for training (pixel_values, image_grid_thw, etc.)
    # Mirrors geo3k_vlm_multi_turn/rollout.py _encode_observation_for_generation.
    multimodal_train_inputs = {
        k: v for k, v in processor_output.items()
        if k not in ["input_ids", "attention_mask"]
    } or None

    return token_ids, pil_images, multimodal_train_inputs

# ---------------------------------------------------------------------------
# Sample data saving utilities
# ---------------------------------------------------------------------------
def _to_serializable(obj):
    """Convert numpy/torch types and custom objects to native Python types."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, type):
        return str(obj)
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    if hasattr(obj, 'item') and not isinstance(obj, dict):
        try:
            return obj.item()
        except (TypeError, ValueError):
            pass
    if hasattr(obj, 'model_dump') and callable(getattr(obj, 'model_dump')):
        try:
            return _to_serializable(obj.model_dump())
        except TypeError:
            pass
    if hasattr(obj, '__dataclass_fields__'):
        from dataclasses import asdict
        return _to_serializable(asdict(obj))
    if isinstance(obj, list):
        return [_to_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, Enum):
        return obj.value
    return str(obj)


def serialize_sample(sample: Sample, rollout_idx: int=-1, keep_rollout_details: bool=False) -> dict:
    """Serialize a Sample object to a flat dict suitable for HuggingFace Dataset."""
    res = {
        "rollout_idx": rollout_idx,
        "group_index": sample.group_index,
        "index": sample.index,
        "prompt": sample.prompt,
        # "multimodal_inputs": _to_serializable(sample.multimodal_inputs), # 
        "response": sample.response,
        "response_length": sample.response_length,
        "reward": _to_serializable(sample.reward),
        "status": _to_serializable(sample.status),
        "metadata": _to_serializable(sample.metadata),
    }

    if keep_rollout_details:
        res.update({
            "tokens": _to_serializable(sample.tokens) or [],
            "loss_mask": _to_serializable(sample.loss_mask) or [],
            "rollout_log_probs": _to_serializable(sample.rollout_log_probs) or [],
        })

    return res

def _save_sample(samples_dir: str, sample_id: str, sample: Sample, messages: list[dict] = [], keep_rollout_details: bool = False) -> None:
    os.makedirs(samples_dir, exist_ok=True)

    serialized_sample = serialize_sample(sample, keep_rollout_details=keep_rollout_details)

    path_to_save = os.path.join(samples_dir, f"{sample_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{sample.status.value}_{sample.metadata['total_steps']}steps.json")
    with open(path_to_save, "w") as f:
        json.dump(serialized_sample, f, indent=2, ensure_ascii=False)

def _save_per_token_dump(sample: Sample, tokenizer, path: str) -> None:
    """Write a per-token debug file for one (turn) sample.

    One line per token: ``decoded<TAB>loss_mask<TAB>logprob<TAB>token_id``.
    ``loss_mask`` / ``rollout_log_probs`` cover only the response portion, so the leading prompt tokens are written with
    ``-`` for mask and logprob. Tab / newline characters inside a decoded token are escaped so each token stays on exactly one line.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    prompt_len = len(sample.tokens) - len(sample.loss_mask)
    with open(path, "w") as f:
        for i, token_id in enumerate(sample.tokens):
            decoded = (
                tokenizer.decode([token_id])
                .replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")
            )
            if i < prompt_len:
                f.write(f"{decoded}\t-\t-\t{token_id}\n")
            else:
                ri = i - prompt_len
                f.write(f"{decoded}\t{sample.loss_mask[ri]}\t{sample.rollout_log_probs[ri]}\t{token_id}\n")

# ---------------------------------------------------------------------------
# Main generate function
# ---------------------------------------------------------------------------

def _finalize_turn_samples(
    turn_samples: list[Sample],
    status: Sample.Status,
    terminate_reason: str,
    total_steps: int,
    images: list[str] = []
):
    """Mark all turn samples with final status, reason, and step count."""
    if turn_samples:
        turn_samples[-1].metadata["is_last_turn"] = True
        turn_samples[-1].metadata["full_image_list"] = images
    for ts in turn_samples:
        ts.status = status
        ts.metadata["terminate_reason"] = terminate_reason
        ts.metadata["total_steps"] = total_steps
        # Number of turn-samples in this trajectory, for equal per-turn loss
        # weighting under --turn-level-loss-equal-turn-weight.
        ts.metadata["num_turns_in_trajectory"] = len(turn_samples)
        # remove_sample zeroes the loss mask downstream (slime/ray/rollout.py).
        # reward=0.0 MUST be set here: slime short-circuits the reward model for any turn-list containing 
        # an ABORTED sample (sglang_rollout.py: `if any(... ABORTED): return samples`), so reward_func 
        # never runs on it and the reward would otherwise stay None and crash GRPO normalization.
        if status == Sample.Status.ABORTED:
            ts.remove_sample = True
            ts.reward = 0.0


def _make_aborted_empty_turn_sample(
    source_sample: Sample,
    task_data: dict,
    messages: list[dict],
    images: list[str],
    terminate_reason: str,
) -> Sample:
    """Return a training-safe placeholder when generation fails before turn 0."""
    metadata = dict(task_data or {})
    metadata.update(
        {
            "turn_index": 0,
            "is_last_turn": True,
            "messages": list(messages),
            "images": list(images),
            "full_image_list": list(images),
            "terminate_reason": terminate_reason,
            "total_steps": 0,
            "num_turns_in_trajectory": 1,
        }
    )

    # Dataset prompt samples do not necessarily carry token IDs. Keep one
    # masked prompt token so downstream packers never see an empty sequence.
    tokens = list(source_sample.tokens or [0])

    return Sample(
        index=source_sample.index,
        rollout_id=source_sample.index,
        group_index=source_sample.group_index,
        prompt=source_sample.prompt,
        label=source_sample.label,
        tokens=tokens,
        response="",
        response_length=0,
        loss_mask=[],
        rollout_log_probs=[],
        reward=0.0,
        remove_sample=True,
        status=Sample.Status.ABORTED,
        metadata=metadata,
        multimodal_inputs=source_sample.multimodal_inputs,
        multimodal_train_inputs=source_sample.multimodal_train_inputs,
        custom_rm_path=source_sample.custom_rm_path,
        generate_function_path=source_sample.generate_function_path,
    )


async def _safe_exit_env(env: Any, task_id: str, timeout_secs: float) -> None:
    """Tear down the env, shielding cleanup so a slow or cancelled rollout can't leak it.

    A plain ``await env.exit()`` would be cancelled along with the rollout (e.g. when the
    per-task timeout below fires), leaving the sandbox pod / remote session leaked.
    ``asyncio.shield`` lets the teardown finish in the background instead.
    """
    cleanup = asyncio.shield(asyncio.ensure_future(env.exit()))
    try:
        await asyncio.wait_for(cleanup, timeout=timeout_secs)
        logger.info(f"Task {task_id}: environment exited successfully.")
    except asyncio.TimeoutError:
        logger.warning(f"Task {task_id}: environment exit timed out after {timeout_secs}s; detaching cleanup.")
    except Exception as e:
        logger.warning(f"Task {task_id}: environment exit failed: {e}")


async def generate_trajectory_sample(args: Any, sample: Sample, sampling_params: dict) -> list[Sample]:
    # Trajectory-level (single-stream) generation is intentionally not supported here;
    # delegate to the turn-level path so both entrypoints behave identically.
    return await generate_turn_sample(args, sample, sampling_params)


async def generate_turn_sample(
    args: Any,
    sample: Sample,
    sampling_params: dict,
) -> list[Sample]:
    """Run the turn-level rollout under an optional overall timeout.

    The rollout logic lives in ``_generate_turn_sample_impl``; this wrapper only adds a
    best-effort wall-clock cap (``rollout_task_timeout_secs``, disabled by default) on top
    of the per-await timeouts inside it. A multi-turn rollout can otherwise run for
    ``max_steps`` × (inference + env_step) seconds before any single await trips, so a
    stuck env could hold a rollout slot for a very long time. On timeout the rollout is
    cancelled (the shielded ``_safe_exit_env`` still tears down the env) and a cleanly
    removed ABORTED sample is returned.
    """
    timeout_secs = getattr(args, "rollout_task_timeout_secs", None)
    if not timeout_secs:
        return await _generate_turn_sample_impl(args, sample, sampling_params)

    task_id = str((sample.metadata or {}).get("task_id", "unknown"))
    try:
        return await asyncio.wait_for(
            _generate_turn_sample_impl(args, sample, sampling_params),
            timeout=float(timeout_secs),
        )
    except asyncio.TimeoutError:
        logger.warning(f"Task {task_id}: rollout timed out after {timeout_secs}s; aborting.")
        return [
            _make_aborted_empty_turn_sample(
                sample,
                sample.metadata or {},
                messages=[],
                images=[],
                terminate_reason=f"generation_error: rollout_task_timeout after {timeout_secs}s",
            )
        ]


async def _generate_turn_sample_impl(
    args: Any,
    sample: Sample,
    sampling_params: dict,
) -> list[Sample]:
    """
    Generate a multi-turn agent-environment interaction, returning one Sample per turn.

    Each turn sample contains its own prompt (full context up to that turn) and
    response (the LLM output for that turn). Only LLM response tokens have loss_mask=1.

    Args:
        args: Rollout arguments from slime training pipeline
        sample: Sample containing task info
        sampling_params: LLM sampling parameters

    Returns:
        List of per-turn Sample objects.
    """
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)

    assert not getattr(args, "partial_rollout", False), \
        "Partial rollout is not supported for gym environment interactions."

    context_num_screenshots = getattr(args, "context_num_screenshots", 1)
    # Per-await timeouts (seconds), configured in browser_training_config.yaml.
    # On timeout asyncio.wait_for raises TimeoutError (a subclass of Exception),
    # which the existing handlers below turn into a cleanly ABORTED sample.
    env_init_timeout = getattr(args, "env_init_timeout", 300)   # sandbox create + reset
    inference_timeout = getattr(args, "inference_timeout", 300)  # one SGLang /generate
    env_step_timeout = getattr(args, "env_step_timeout", 120)    # one browser action (~20s normal)
    env_exit_timeout = getattr(args, "env_exit_timeout", 60)     # sandbox teardown

    task_data = sample.metadata or {}
    task_id = task_data.get("task_id", "unknown")
    sample_id = str(task_id).replace('/', '_').replace('--', '_')
    user_request = task_data.get("intent", "")
    print(f"Generating turn data for sample {task_id}: {user_request}")

    messages: list[dict] = []
    images: list[str] = []
    turn_samples: list[Sample] = []
    env = None  # initialized early so finally block can always clean up

    try:
        env, adapter, state, url = await asyncio.wait_for(
            _initialize_resources(args, task_data), timeout=env_init_timeout
        )
        observation, info = await asyncio.wait_for(env.reset(), timeout=env_init_timeout)

        tokenizer = state.tokenizer
        processor = state.processor
        sampling_params = sampling_params.copy()
        tools_info, policy = adapter.parse_env_info(info)

        adapter._tools_info = tools_info
        parser_type = resolve_parser_type(tokenizer.name_or_path)
        adapter._tool_parser = ToolParser(tools_info, parser_type=parser_type)
        
        _SCREENSHOT_PLACEHOLDER = "screenshot:\n<|vision_start|><|image_pad|><|vision_end|>\n"

        # --- Initialize message tracking ---
        initial_messages, initial_image = adapter.get_initial_messages(user_request, policy, observation)
        input_text = adapter.token_handler.apply_chat_template(
            initial_messages, tools_info, add_generation_prompt=True
        )
        
        messages.extend(initial_messages)
        images.append(initial_image)

        # --- Main interaction loop ---
        terminated = False
        consecutive_parse_failures = 0

        for step in range(getattr(args, "max_steps", 16)):
            # If there are more than context_num_screenshots placeholders, keep only the latest ones and remove earlier ones.
            placeholder_count = input_text.count(_SCREENSHOT_PLACEHOLDER)
            if placeholder_count > context_num_screenshots:
                parts = input_text.split(_SCREENSHOT_PLACEHOLDER)
                n_remove = placeholder_count - context_num_screenshots
                prefix = "".join(parts[:n_remove + 1])
                suffix = _SCREENSHOT_PLACEHOLDER.join(parts[n_remove + 1:])
                input_text = prefix + _SCREENSHOT_PLACEHOLDER + suffix

            # Compute input tokens with image_pad correctly expanded
            img_list = images[-context_num_screenshots:]
            input_tokens, pil_images, mm_train = _encode_with_processor(
                processor, tokenizer, input_text, img_list
            )
            # Check max context length
            max_ctx_len = getattr(args, "rollout_max_context_len", None)
            if max_ctx_len and (len(input_tokens) > max_ctx_len):
                logger.info(f"Step {step}: turn input length {len(input_tokens)} exceeds max_context_len {max_ctx_len}, stopping.")
                _finalize_turn_samples(turn_samples, Sample.Status.TRUNCATED, "generation_length_limit", step + 1)
                break

            # Create turn sample
            turn_sample = Sample(
                index=sample.index,
                # All per-turn samples from one trajectory share rollout_id so slime's loss reducer aggregates them as 
                # a single rollout (not N). See _validate_rollout_id_annotated in slime/ray/rollout.py.
                rollout_id=sample.index,
                # Preserve the prompt-group from the data source (shared by the N
                # trajectories sampled for this prompt) so GRPO normalizes per
                # prompt-group, not per turn. The turn index lives in
                # metadata["turn_index"] below.
                group_index=sample.group_index,
                prompt=input_text,
                response_length=0,
                tokens=input_tokens,
                loss_mask=[],
                rollout_log_probs=[],
                metadata=dict(task_data),
            )
            # Key "turn_index" and "is_last_turn" are used for the indicator of turn-level samples.
            turn_sample.metadata["turn_index"] = step
            turn_sample.metadata["is_last_turn"] = False
            turn_sample.multimodal_inputs = {"images": pil_images}
            turn_sample.multimodal_train_inputs = mm_train

            # Run inference ---------------------------------------------
            llm_response, new_tokens, new_logprobs, finish_type = await asyncio.wait_for(
                _run_inference_step(url, input_tokens, sampling_params, img_list, tokenizer),
                timeout=inference_timeout,
            )
            # --------------------------------------------------------------
            # print(llm_response)

            # Append LLM response tokens (loss_mask=1)
            _append_to_sample(turn_sample, new_tokens, new_logprobs, loss_mask_val=1)

            turn_sample.response = llm_response
            turn_sample.response_length = len(new_tokens)            
            _ensure_correct_sample(turn_sample, tokenizer) # for debugging: verify final token stream is correct after merging multimodal inputs
            turn_samples.append(turn_sample)

            input_text += llm_response
            llm_response_wo_end = adapter.preprocess_response(llm_response) # remove "<|im_end|>"
            # messages.append({"role": "assistant", "content": llm_response_wo_end})
            stored_assistant = llm_response_wo_end
            if not args.include_history_thoughts and "</think>" in stored_assistant:
                stored_assistant = stored_assistant.split("</think>", 1)[-1].lstrip()
            messages.append({"role": "assistant", "content": stored_assistant})

            turn_sample.metadata["images"] = img_list
            turn_sample.metadata["messages"] = messages.copy()  # capture messages up to this turn

            # Check if generation should stop (from the SGLang side)
            if finish_type == "length":
                _finalize_turn_samples(turn_samples, Sample.Status.TRUNCATED, "generation_length_limit", step + 1)
                break
            if finish_type == "abort":
                _finalize_turn_samples(turn_samples, Sample.Status.ABORTED, "generation_abort", step + 1)
                break

            # Parse LLM_response to execute action
            parsed = adapter.tool_parser.parse(llm_response_wo_end)
            if parsed.success and len(parsed.calls) > 0:
                actions = adapter.actions_from_parsed(parsed.calls)

                try:
                    observation, _, terminated, _, info = await asyncio.wait_for(
                        env.step(actions), timeout=env_step_timeout
                    )
                except Exception as step_err:
                    logger.warning(f"Task {task_id} step {step}: env.step() failed, marking as ABORTED: {step_err}")
                    # raise ValueError(f"❗  Environment step failed: {step_err}") from step_err
                    _finalize_turn_samples(turn_samples, Sample.Status.ABORTED, "env_step_error", step + 1)
                    break

                tool_msgs = [{
                    "role": "tool",
                    "name": tool_result["tool_name"],
                    "content": adapter.clean_tool_response(tool_result["tool_response"]),
                } for tool_result in info["tool_responses"]]
    
                consecutive_parse_failures = 0  # reset on success
                
                # Check if the agent called done
                if terminated:
                    _finalize_turn_samples(turn_samples, Sample.Status.COMPLETED, "task_completed", step + 1, images) # include full image list for completed sample to compute reward
                    break

            else:
                print(f"⚠️  Failed to parse tool calls from the response. {llm_response}")
                consecutive_parse_failures += 1
                if consecutive_parse_failures >= getattr(args, "max_consecutive_parse_failures", 3):
                    logger.warning(f"Task {task_id} step {step}: {consecutive_parse_failures} consecutive parse failures, aborting")
                    _finalize_turn_samples(turn_samples, Sample.Status.FAILED, "consecutive_parse_failures", step + 1)
                    break
                tool_msgs = [{"role": "tool", "content": "Failed to parse tool calls from the response. Please try again with correct format."}]

            # append observation to tool response messages
            ob_text, ob_screen_base64 = adapter.render_observation(observation)
            if not args.include_tool_response:
                tool_msgs = [{"role": "user", "content": ""}] # used for ablation on tool response
                tool_msgs[-1]["content"] += f"{ob_text}" # used for ablation on tool response
            else:
                tool_msgs[-1]["content"] += f"\n{ob_text}"
            if ob_screen_base64:
                images.append(ob_screen_base64)

            messages.extend(tool_msgs)
            full_text = adapter.token_handler.apply_chat_template(
                messages, tools_info, add_generation_prompt=True
            )
            input_text = full_text
        
        # --- Finalization (only if loop exhausted without break) ---
        else:
            _finalize_turn_samples(turn_samples, Sample.Status.FAILED, "max_steps_exhausted", getattr(args, "max_steps", 16))
        
        return turn_samples

    except Exception as e:
        logger.opt(exception=True).warning("Task {}: generate_turn_sample failed: {}", task_id, e)
        # raise ValueError(f"❗  Generation failed: {e}") from e
        # Return a failed sample instead of crashing the whole batch.
        if turn_samples:
            # set all turns to ABORTED to avoid introducing noisy tokens
            _finalize_turn_samples(turn_samples, Sample.Status.ABORTED, f"generation_error: {e}", len(turn_samples))
        else:
            turn_samples = [
                _make_aborted_empty_turn_sample(
                    sample,
                    task_data,
                    messages,
                    images,
                    terminate_reason=f"generation_error: {e}",
                )
            ]
        return turn_samples
    finally:
        try:
            path_to_save_generated_samples = getattr(args, "path_to_save_generated_samples", "")
            if path_to_save_generated_samples and messages:
                _save_sample(
                    samples_dir=os.path.join(path_to_save_generated_samples, "samples"), # , _EXPT_ID),
                    sample_id=sample_id,
                    sample=turn_samples[-1],
                )
        except Exception as save_err:
            logger.warning(f"Task {task_id}: failed to save sample data: {save_err}")

        if env is not None:
            await _safe_exit_env(env, task_id, env_exit_timeout)
