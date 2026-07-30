import asyncio
import logging

import aiohttp
import numpy as np
import torch

from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.types import Sample

# Compact dtypes for shipping teacher top-k through Ray (vs Python lists): float16
# logprobs + int32 ids are ~7-16x smaller in RAM and serialize zero-copy via Arrow.
# float16 holds logprobs (~[-20,0]) fine; the pad sentinel below stays float16-finite.
_TOPK_LP_DTYPE = np.float16
_TOPK_ID_DTYPE = np.int32

logger = logging.getLogger(__name__)

# Cached tokenizers — initialised lazily in reward_func
_student_tokenizer = None
_teacher_tokenizer = None
_cross_vocab: bool | None = None  # None = not yet determined
_resync_pairs: list | None = None  # cached resync pairs for cross-vocab alignment
_teacher_max_len: int | None = None  # cached from server when not set in config

# Fallback timeout when no config is loaded
_DEFAULT_TEACHER_TIMEOUT = 300  # seconds


async def _fetch_teacher_max_len(url: str, timeout: float) -> int | None:
    """Discover the teacher's maximum accepted *input* length.

    Prefers ``max_req_input_len`` from SGLang's ``/get_server_info`` — this is the
    real hard ceiling the server enforces on ``input_ids``. NOTE: the check is
    inclusive — SGLang rejects any input whose length is ``>= max_req_input_len``
    (the maximum *accepted* length is ``max_req_input_len - 1``), so callers must
    keep the sent sequence strictly below this value. Falls back to
    ``max_model_len`` from ``/v1/models`` (the context window, which is a few
    tokens larger than the input cap) when ``/get_server_info`` is unavailable.

    Returns None if both requests fail or the fields are absent. The base URL is
    derived from the generate endpoint, e.g.
    'http://host:port/generate' -> 'http://host:port/get_server_info'.
    """
    from urllib.parse import urlparse, urlunparse
    parsed = urlparse(url)

    # Preferred: the server's enforced input-length cap.
    info_url = urlunparse(parsed._replace(path="/get_server_info"))
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as session:
            async with session.get(info_url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    max_input = data.get("max_req_input_len")
                    if max_input is not None:
                        logger.info(
                            f"[opd] auto-discovered teacher_max_len={max_input} "
                            f"(max_req_input_len) from {info_url}"
                        )
                        return int(max_input)
    except Exception as e:
        logger.warning(f"[opd] could not fetch max_req_input_len from {info_url}: {e}")

    # Fallback: the model context window from the OpenAI-compatible endpoint.
    models_url = urlunparse(parsed._replace(path="/v1/models"))
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as session:
            async with session.get(models_url) as resp:
                resp.raise_for_status()
                data = await resp.json()
                max_len = data["data"][0].get("max_model_len")
                if max_len is not None:
                    logger.info(f"[opd] auto-discovered teacher_max_len={max_len} from {models_url}")
                return max_len
    except Exception as e:
        logger.warning(f"[opd] could not fetch teacher_max_len from {models_url}: {e}")
        return None


def _build_resync_pairs(student_tokenizer, teacher_tokenizer, token_strings: list[str]) -> list[tuple[int, int]]:
    """
    Build (student_token_id, teacher_token_id) resync anchor pairs from a list
    of token strings that are shared identically between both tokenizers.

    Tokens that map to UNK in either tokenizer are silently skipped.
    """
    pairs = []
    for tok_str in token_strings:
        s_id = student_tokenizer.convert_tokens_to_ids(tok_str)
        t_id = teacher_tokenizer.convert_tokens_to_ids(tok_str)
        if s_id == student_tokenizer.unk_token_id or t_id == teacher_tokenizer.unk_token_id:
            continue
        pairs.append((s_id, t_id))
    logger.info(f"[opd] Built {len(pairs)} resync pairs for cross-vocab alignment: {pairs}")
    return pairs


def _load_config(args):
    """Return an OpdConfig, loading from --opd-config YAML if provided.

    Falls back to constructing one from legacy args (rm_url, rm_tokenizer_path,
    opd_kl_coef) so existing training scripts continue to work unchanged.
    """
    from slime.rollout.opd_config import OpdConfig, load_opd_config

    opd_config_path = getattr(args, "opd_config", None)
    if opd_config_path is not None:
        return load_opd_config(opd_config_path)

    # Legacy fallback: build config from individual args
    url = getattr(args, "rm_url", None) or ""
    tokenizer_path = getattr(args, "rm_tokenizer_path", None) or None
    kl_coef = float(getattr(args, "opd_kl_coef", 1.0))
    return OpdConfig(url=url, tokenizer_path=tokenizer_path, kl_coef=kl_coef)


def _is_same_vocab(student_tokenizer, teacher_tokenizer) -> bool:
    """Return True if both tokenizers share the same vocabulary."""
    if len(student_tokenizer) != len(teacher_tokenizer):
        return False
    _CHECK_IDS = list(range(0, min(100, len(student_tokenizer), len(teacher_tokenizer))))
    return all(
        teacher_tokenizer.convert_ids_to_tokens(i) == student_tokenizer.convert_ids_to_tokens(i)
        for i in _CHECK_IDS
    )


def _truncate_to_teacher_max_len(
    token_ids: list[int],
    teacher_max_len: int,
    response_length: int,
    truncation_side: str = "prefix",
) -> tuple[list[int], int, int]:
    """Clamp *token_ids* so the teacher accepts the request.

    Ensures ``len(returned_ids) <= teacher_max_len``. The caller is responsible
    for passing a ``teacher_max_len`` strictly below the server's
    ``max_req_input_len`` — SGLang rejects any input whose length is
    ``>= max_req_input_len`` with HTTP 400 (the max accepted length is
    ``max_req_input_len - 1``). The bound is on the *total* sequence
    (prompt + response).

    Simple end-truncation by *truncation_side*:

      - "prefix" (default): drop ``overflow`` tokens from the left. Since the
        response sits at the tail, this normally removes only early prompt
        context and keeps every response log-prob valid.
      - "suffix": drop ``overflow`` tokens from the right (response tail first).
        Dropped response positions get teacher_log_prob=0 / valid_mask=0.

    Returns:
        (truncated_ids, n_prompt_dropped, n_response_dropped) — the trimmed
        sequence and how many prompt/response tokens were removed. Only
        ``n_response_dropped`` affects the response valid_mask.
    """
    n = len(token_ids)
    if n <= teacher_max_len:
        return token_ids, 0, 0

    overflow = n - teacher_max_len
    prompt_length = n - response_length

    if truncation_side == "suffix":
        # Drop from the right; the rightmost tokens are response tokens.
        truncated = token_ids[: n - overflow]
        n_response_dropped = min(overflow, response_length)
        n_prompt_dropped = overflow - n_response_dropped
    else:  # "prefix"
        # Drop from the left; the leftmost tokens are prompt tokens.
        truncated = token_ids[overflow:]
        n_prompt_dropped = min(overflow, prompt_length)
        n_response_dropped = overflow - n_prompt_dropped

    return truncated, n_prompt_dropped, n_response_dropped


async def reward_func(args, sample, **kwargs):
    """Query teacher and return aligned log-probs for KL distillation.

    Always returns a dict: {"teacher_log_probs": list[float], "valid_mask": list[int]}.
    Both cross-vocab and same-vocab paths produce this unified shape.
    """
    global _student_tokenizer, _teacher_tokenizer, _cross_vocab, _resync_pairs, _teacher_max_len

    cfg = _load_config(args)

    # Lazy-fetch teacher_max_len from server when not explicitly configured (once per worker).
    if cfg.teacher_max_len is None and _teacher_max_len is None:
        logger.warning(
            "[opd] teacher_max_len is not set in config — querying teacher server for max_model_len. "
            "Set teacher_max_len explicitly in your OPD config to suppress this warning."
        )
        timeout = getattr(cfg, "timeout", None) or _DEFAULT_TEACHER_TIMEOUT
        fetched = await _fetch_teacher_max_len(cfg.url, timeout)
        # max_req_input_len is an *inclusive* reject threshold (SGLang refuses
        # length >= max_req_input_len), so the max accepted length is one less.
        # Subtract 1 here so truncation never produces the boundary value that
        # would 400. 0 = fetch attempted, nothing found.
        _teacher_max_len = (fetched - 1) if fetched else 0
    effective_max_len = cfg.teacher_max_len or (_teacher_max_len if _teacher_max_len else None)


    # Same-vocab: query teacher for raw log-probs and extract inline
    teacher_max_len = effective_max_len
    teacher_truncation_side = getattr(cfg, "teacher_truncation_side", "prefix")

    # "prefix" (default): drop prompt-prefix tokens so the full response tail is
    #   kept — every response token still receives a valid teacher log-prob.
    # "suffix": drop response-tail tokens beyond the cap; those positions get
    #   teacher_log_prob=0 / valid_mask=0.
    # In both cases we guarantee len(input_ids) <= teacher_max_len so the server
    # never rejects the request with HTTP 400 (input longer than max_req_input_len).
    input_ids = list(sample.tokens)
    n_response_dropped = 0
    if teacher_max_len is not None and len(input_ids) > teacher_max_len:
        original_len = len(input_ids)
        input_ids, n_prompt_dropped, n_response_dropped = _truncate_to_teacher_max_len(
            input_ids, teacher_max_len, sample.response_length, teacher_truncation_side
        )
        logger.warning(
            f"[opd] same-vocab: sample.tokens length {original_len} exceeds "
            f"teacher_max_len={teacher_max_len} (side={teacher_truncation_side}) — "
            f"kept {len(input_ids)} tokens ({n_prompt_dropped} prompt + "
            f"{n_response_dropped} response tokens dropped)"
        )

    # Only request logprobs for the RESPONSE positions, not the whole prompt.
    # SGLang returns logprobs for input positions >= logprob_start_len, with the
    # first returned position carrying a None logprob (the boundary). The response
    # tokens still present in the (possibly truncated) sent sequence sit at its
    # tail, so starting one token before them yields exactly the response logprobs
    # after the standard [1:] skip. This shrinks the teacher response by ~k*(prompt/
    # response) — critical with top_logprobs_num > 1, where returning top-k for the
    # entire ~75k-token prompt produces a multi-MB payload that destabilises the
    # rollout node at the colocate hand-off. logprob_start_len does NOT change the
    # logprob VALUES (the full sequence is always sent as context).
    n_response_sent = sample.response_length - n_response_dropped
    logprob_start_len = max(0, len(input_ids) - n_response_sent - 1)

    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 1.0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": logprob_start_len,
    }
    timeout = getattr(cfg, "timeout", None) or _DEFAULT_TEACHER_TIMEOUT
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as session:
            async with session.post(cfg.url, json=payload) as resp:
                resp.raise_for_status()
                response = await resp.json()
    except asyncio.TimeoutError:
        logger.warning(f"[opd] Same-vocab teacher at {cfg.url!r} timed out after {timeout}s")
        raise
    except Exception as e:
        logger.warning(f"[opd] Same-vocab teacher at {cfg.url!r} error: {e}")
        raise

    response_length = sample.response_length
    raw_lps = _extract_token_logprobs(response)  # covers sent tokens[1:]

    # The kept response tokens are a contiguous block in the *sent* sequence's
    # tail, so their log-probs are the last entries of raw_lps. Where the dropped
    # response tokens sit within the full response depends on the truncation side:
    #   suffix: dropped from the right -> kept response is the HEAD, zeros go last
    #   prefix: dropped from the left  -> kept response is the TAIL, zeros go first
    # Build the full response_length arrays accordingly.
    n_kept = response_length - n_response_dropped          # response tokens still sent
    n_valid = min(n_kept, len(raw_lps))                    # log-probs we actually got
    n_front_pad = n_kept - n_valid                         # first kept token has no lp (sent[0])
    if n_front_pad > 0:
        logger.warning(
            f"[opd] same-vocab: teacher returned {len(raw_lps)} log-probs but "
            f"expected {n_kept} for the response — front-padding {n_front_pad} "
            f"positions with zeros (valid_mask=0)"
        )
    kept_lps = raw_lps[-n_valid:] if n_valid > 0 else []
    if teacher_truncation_side == "prefix":
        # Dropped response tokens (if any) are the HEAD of the response.
        teacher_lps = [0.0] * (n_response_dropped + n_front_pad) + kept_lps
        valid_mask = [0] * (n_response_dropped + n_front_pad) + [1] * n_valid
    else:  # suffix: dropped response tokens are the TAIL of the response.
        teacher_lps = [0.0] * n_front_pad + kept_lps + [0.0] * n_response_dropped
        valid_mask = [0] * n_front_pad + [1] * n_valid + [0] * n_response_dropped
    # Tripwire: these arrays are consumed token-for-token against
    # student_log_probs / loss_mask in apply_opd_kl_to_advantages, so a length
    # other than response_length would crash or silently misalign the KL.
    assert len(teacher_lps) == response_length == len(valid_mask), (
        f"[opd] same-vocab teacher_log_probs length {len(teacher_lps)} / "
        f"valid_mask length {len(valid_mask)} != response_length {response_length}"
    )
    result = {"teacher_log_probs": teacher_lps, "valid_mask": valid_mask, "reward": 0.0}

    return result


async def _query_teacher(
    session: aiohttp.ClientSession,
    url: str,
    token_ids: list[int],
    timeout: float,
) -> dict | None:
    """Query a sglang teacher server with pre-tokenized token IDs."""
    payload = {
        "input_ids": token_ids,
        "sampling_params": {
            "temperature": 1.0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }

    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        image_data = sample.multimodal_inputs["images"]
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in image_data]

    session_kwargs = {}
    async with aiohttp.ClientSession(**session_kwargs) as session:
        async with session.post(args.rm_url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


def _extract_token_logprobs(response: dict) -> list[float]:
    """Extract per-token log-probs from a sglang response."""
    return [item[0] for item in response["meta_info"]["input_token_logprobs"][1:]]


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Extract teacher log-probs from reward_func results and set Sample fields.

    Both cross-vocab and same-vocab reward_func paths return the same dict:
      {"teacher_log_probs": list[float], "valid_mask": list[int]}
    Optionally also:
      {"teacher_topk_logprobs": list[float], "teacher_topk_ids": list[int],
       "teacher_topk_rank": list[int]}
    We simply unpack it here and zero out sample.reward.

    Returns scalar rewards (0.0) — learning signal comes from OPD KL penalty.
    """
    cfg = _load_config(args)


    for sample in samples:
        d = sample.reward
        sample.teacher_log_probs = torch.tensor(d["teacher_log_probs"], dtype=torch.float32)
        valid_mask = d["valid_mask"]
        sample.teacher_valid_mask = valid_mask
        sample.reward = 0.0

    scalar_rewards = [0.0] * len(samples)

    return scalar_rewards, scalar_rewards
