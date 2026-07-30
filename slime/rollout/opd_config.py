# slime/rollout/opd_config.py
"""
Config dataclass and YAML loader for single-teacher on-policy distillation (OPD).

Config file format (YAML):

    teacher:
      url: "http://teacher:30001/generate"
      tokenizer_path: "/models/DeepSeek-V3"   # omit or null for same-vocab teacher

    kl_coef: 1.0          # KL penalty coefficient applied to advantages
    timeout: 120          # seconds to wait for teacher server response
    teacher_topk: 1       # if > 1, query teacher for top-k token log-probs per position
                          # and populate sample.teacher_topk_logprobs / teacher_topk_ids /
                          # teacher_topk_rank (only supported for same-vocab sglang path)
    teacher_max_len: 32768        # (optional) max tokens the teacher can accept.
                                  # Behaviour depends on teacher_truncation_side.
    teacher_truncation_side: prefix  # "prefix" (default): trim tokens from the left;
                                     #   this normally drops prompt-prefix tokens and
                                     #   keeps valid log-probs for the response tail.
                                     # "suffix": trim tokens from the right; dropped
                                     #   response tokens get teacher_log_prob=0 and
                                     #   valid_mask=0 (zero KL contribution).

Supported api_type values:

    "sglang"        — (default) SGLang native /generate endpoint
    "openai"        — OpenAI-compatible /v1/completions (vLLM, SGLang OAI, Fireworks, …)
    "azure_openai"  — Azure OpenAI with Azure AD authentication

Examples:

    # Azure OpenAI (TRAPI)
    teacher:
      url: "https://trapi.research.microsoft.com/redmond/interactive/"
      api_type: "azure_openai"
      model: "Qwen/Qwen3.5-397B-A17B-GPTQ-Int4"
      tokenizer_path: "/models/Qwen3.5-4B"
      azure_ad_scope: "api://trapi/.default"
      api_version: "2025-04-01-preview"

    # Fireworks / generic OpenAI-compatible
    teacher:
      url: "https://api.fireworks.ai/inference/v1"
      api_type: "openai"
      model: "accounts/fireworks/models/qwen3p5-397b-a17b"
      api_key: "fw_xxx"
      tokenizer_path: "/models/Qwen3.5-397B"
"""

from __future__ import annotations

from dataclasses import dataclass

import yaml

_VALID_API_TYPES = ("sglang", "openai", "azure_openai")


@dataclass
class OpdConfig:
    url: str                              # teacher server URL
    tokenizer_path: str | None            # None => same vocab as student (fast path)
    kl_coef: float = 1.0                  # KL penalty coefficient
    timeout: float = 120.0                # HTTP timeout in seconds
    api_type: str = "sglang"              # "sglang" | "openai" | "azure_openai"
    model: str | None = None              # model/deployment name (required for openai/azure_openai)
    api_key: str | None = None            # API key for openai api_type
    api_version: str | None = None        # Azure API version (azure_openai only)
    azure_ad_scope: str | None = None     # Azure AD token scope (azure_openai only)
    teacher_topk: int = 1                 # if > 1, query teacher for top-k token log-probs and
                                          # populate teacher_topk_logprobs/ids/rank on Sample
                                          # (same-vocab sglang path only)
    teacher_max_len: int | None = None    # maximum number of tokens the teacher can accept.
                                          # When set and the sequence exceeds this limit,
                                          # truncation is applied per teacher_truncation_side.
    teacher_truncation_side: str = "suffix"
                                          # "prefix" (default): trim tokens from the left;
                                          #   this normally drops prompt-prefix tokens and keeps
                                          #   valid log-probs for the response tail (valid_mask=1).
                                          # "suffix": trim tokens from the right; dropped
                                          #   response tokens get teacher_log_prob=0 and
                                          #   valid_mask=0 (zero KL contribution).


def load_opd_config(path: str) -> OpdConfig:
    """Load and validate a single-teacher OPD config from a YAML file.

    Raises:
        ValueError: If required fields are missing or invalid.
        FileNotFoundError: If the path does not exist.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)

    teacher = raw.get("teacher")
    if not teacher:
        raise ValueError(f"Config at {path!r} must have a 'teacher' section.")

    url = teacher.get("url")
    if not url:
        raise ValueError(f"Config at {path!r}: teacher.url is required.")

    tokenizer_path = teacher.get("tokenizer_path") or None  # empty string -> None

    kl_coef = float(raw.get("kl_coef", 1.0))
    if kl_coef < 0:
        raise ValueError(f"kl_coef must be >= 0, got {kl_coef}.")

    timeout = float(raw.get("timeout", 120.0))

    api_type = teacher.get("api_type", "sglang")
    if api_type not in _VALID_API_TYPES:
        raise ValueError(f"teacher.api_type must be one of {_VALID_API_TYPES}, got {api_type!r}.")

    model = teacher.get("model") or None
    if api_type in ("openai", "azure_openai") and not model:
        raise ValueError(f"teacher.model is required when api_type={api_type!r}.")

    api_key = teacher.get("api_key") or None
    api_version = teacher.get("api_version") or None
    azure_ad_scope = teacher.get("azure_ad_scope") or None

    teacher_topk = int(raw.get("teacher_topk", 1))
    if teacher_topk < 1:
        raise ValueError(f"teacher_topk must be >= 1, got {teacher_topk}.")

    teacher_max_len_raw = raw.get("teacher_max_len", None)
    teacher_max_len = int(teacher_max_len_raw) if teacher_max_len_raw is not None else None
    if teacher_max_len is not None and teacher_max_len < 1:
        raise ValueError(f"teacher_max_len must be >= 1, got {teacher_max_len}.")

    teacher_truncation_side = raw.get("teacher_truncation_side", "prefix")
    if teacher_truncation_side not in ("prefix", "suffix"):
        raise ValueError(
            f"teacher_truncation_side must be 'prefix' or 'suffix', got {teacher_truncation_side!r}."
        )

    return OpdConfig(
        url=url,
        tokenizer_path=tokenizer_path,
        kl_coef=kl_coef,
        timeout=timeout,
        api_type=api_type,
        model=model,
        api_key=api_key,
        api_version=api_version,
        azure_ad_scope=azure_ad_scope,
        teacher_topk=teacher_topk,
        teacher_max_len=teacher_max_len,
        teacher_truncation_side=teacher_truncation_side,
    )
