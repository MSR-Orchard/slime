"""Unit tests for ``_compute_prm_metrics`` from ``slime/ray/rollout.py``.

We load only the function itself by extracting it from rollout.py source
without importing the full module (which needs Ray, torch, sglang, etc.).
The function only depends on numpy, ``dict_add_prefix``, and accesses
``sample.metadata`` — so we use a lightweight stub for Sample.
"""

import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

# --------------------------------------------------------------------------- #
# Load only what _compute_prm_metrics actually needs
# --------------------------------------------------------------------------- #

from slime.utils.metric_utils import dict_add_prefix  # noqa: E402


@dataclass
class _Sample:
    """Minimal Sample stub: only .metadata is used by _compute_prm_metrics."""
    metadata: dict = field(default_factory=dict)


# Build a minimal module namespace and exec just the function from source.
_rollout_src = (Path(__file__).resolve().parents[1] / "slime" / "ray" / "rollout.py").read_text()
_start = _rollout_src.index("def _compute_prm_metrics(")
_end = _rollout_src.index("\ndef _compute_reward_cat_metrics(")
_ns: dict = {"np": np, "dict_add_prefix": dict_add_prefix, "Sample": _Sample, "list": list}
exec(_rollout_src[_start:_end], _ns)  # noqa: S102
_compute_prm_metrics = _ns["_compute_prm_metrics"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sample(prm_dict=None):
    s = _Sample()
    if prm_dict is not None:
        s.metadata["prm"] = prm_dict
    return s


def _success(process_reward=0.8, base=1.0, alpha=0.5, scores=None):
    scores = scores or {"R1": 0.9, "R2": 0.7, "R3": 0.8, "R4": 0.6}
    return {
        "called": True,
        "alpha": alpha,
        "base_binary_reward": base,
        "process_reward": process_reward,
        "final_reward": base + alpha * process_reward,
        "contribution": alpha * process_reward,
        "scores": scores,
        "harmful": False,
        "hallucinated": False,
        "error": "",
    }


def _timeout(base=0.0, alpha=0.5):
    return {
        "called": True,
        "alpha": alpha,
        "base_binary_reward": base,
        "process_reward": 0.0,
        "final_reward": base,
        "contribution": 0.0,
        "scores": {},
        "harmful": False,
        "hallucinated": False,
        "error": "timeout",
    }


def _api_error(base=0.0, alpha=0.5):
    return {
        "called": True,
        "alpha": alpha,
        "base_binary_reward": base,
        "process_reward": 0.0,
        "final_reward": base,
        "contribution": 0.0,
        "scores": {},
        "harmful": False,
        "hallucinated": False,
        "error": "ConnectionError: refused",
    }


def _not_called():
    return {"called": False, "alpha": 0.0}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestComputePrmMetricsBackwardCompat:
    def test_no_prm_key_returns_empty(self):
        """Samples without metadata["prm"] must return {} (backward-compat)."""
        samples = [_Sample(), _Sample()]
        assert _compute_prm_metrics(samples) == {}

    def test_all_alpha_zero_returns_minimal(self):
        """alpha=0 short-circuit sets called=False; API rates should all be 0."""
        samples = [_make_sample(_not_called()), _make_sample(_not_called())]
        m = _compute_prm_metrics(samples)
        assert m["prm/api_called_rate"] == 0.0
        assert m["prm/api_success_rate"] == 0.0
        assert m["prm/api_timeout_rate"] == 0.0
        assert m["prm/api_error_rate"] == 0.0
        # No successful calls → no reward metrics
        assert "prm/process_reward_mean" not in m


class TestApiRates:
    def test_all_success(self):
        samples = [_make_sample(_success()) for _ in range(4)]
        m = _compute_prm_metrics(samples)
        assert m["prm/api_called_rate"] == pytest.approx(1.0)
        assert m["prm/api_success_rate"] == pytest.approx(1.0)
        assert m["prm/api_timeout_rate"] == pytest.approx(0.0)
        assert m["prm/api_error_rate"] == pytest.approx(0.0)

    def test_all_timeout(self):
        samples = [_make_sample(_timeout()) for _ in range(4)]
        m = _compute_prm_metrics(samples)
        assert m["prm/api_called_rate"] == pytest.approx(1.0)
        assert m["prm/api_success_rate"] == pytest.approx(0.0)
        assert m["prm/api_timeout_rate"] == pytest.approx(1.0)
        assert m["prm/api_error_rate"] == pytest.approx(0.0)

    def test_mixed_rates(self):
        # 2 success, 1 timeout, 1 error, 1 not-called (alpha=0)  → n=5
        samples = [
            _make_sample(_success()),
            _make_sample(_success()),
            _make_sample(_timeout()),
            _make_sample(_api_error()),
            _make_sample(_not_called()),
        ]
        m = _compute_prm_metrics(samples)
        n = 5
        assert m["prm/api_called_rate"] == pytest.approx(4 / n)
        assert m["prm/api_success_rate"] == pytest.approx(2 / n)
        assert m["prm/api_timeout_rate"] == pytest.approx(1 / n)
        assert m["prm/api_error_rate"] == pytest.approx(1 / n)


class TestRewardMetrics:
    def test_process_reward_mean_and_median(self):
        rewards = [0.4, 0.6, 0.8, 1.0]
        samples = [_make_sample(_success(process_reward=r)) for r in rewards]
        m = _compute_prm_metrics(samples)
        assert m["prm/process_reward_mean"] == pytest.approx(np.mean(rewards))
        assert m["prm/process_reward_median"] == pytest.approx(np.median(rewards))

    def test_contribution_mean(self):
        alpha = 0.5
        rewards = [0.2, 0.8]
        samples = [_make_sample(_success(process_reward=r, alpha=alpha)) for r in rewards]
        m = _compute_prm_metrics(samples)
        expected = np.mean([alpha * r for r in rewards])
        assert m["prm/contribution_mean"] == pytest.approx(expected)

    def test_alpha_value_propagated(self):
        samples = [_make_sample(_success(alpha=0.3))]
        m = _compute_prm_metrics(samples)
        assert m["prm/alpha"] == pytest.approx(0.3)

    def test_base_reward_mean(self):
        samples = [
            _make_sample(_success(base=1.0)),
            _make_sample(_success(base=0.0)),
        ]
        m = _compute_prm_metrics(samples)
        assert m["prm/base_reward_mean"] == pytest.approx(0.5)


class TestQualityFlags:
    def test_harmful_rate(self):
        d1 = {**_success(), "harmful": True}
        d2 = {**_success(), "harmful": False}
        d3 = {**_success(), "harmful": False}
        samples = [_make_sample(d) for d in [d1, d2, d3]]
        m = _compute_prm_metrics(samples)
        assert m["prm/harmful_rate"] == pytest.approx(1 / 3)

    def test_hallucinated_rate(self):
        d1 = {**_success(), "hallucinated": True}
        d2 = {**_success(), "hallucinated": True}
        d3 = {**_success(), "hallucinated": False}
        samples = [_make_sample(d) for d in [d1, d2, d3]]
        m = _compute_prm_metrics(samples)
        assert m["prm/hallucinated_rate"] == pytest.approx(2 / 3)

    def test_harmful_rate_includes_timeout_calls(self):
        """Harmful flag is checked on ALL called samples, not just successful ones."""
        d_harmful_timeout = {**_timeout(), "harmful": True}
        d_ok = _success()
        samples = [_make_sample(d_harmful_timeout), _make_sample(d_ok)]
        m = _compute_prm_metrics(samples)
        # 1 of 2 called samples is harmful
        assert m["prm/harmful_rate"] == pytest.approx(0.5)


class TestDimMeans:
    def test_all_dims_present(self):
        scores_a = {"R1": 1.0, "R2": 0.8, "R3": 0.6, "R4": 0.4}
        scores_b = {"R1": 0.5, "R2": 0.3, "R3": 0.9, "R4": 0.7}
        samples = [
            _make_sample(_success(scores=scores_a)),
            _make_sample(_success(scores=scores_b)),
        ]
        m = _compute_prm_metrics(samples)
        assert m["prm/R1_mean"] == pytest.approx((1.0 + 0.5) / 2)
        assert m["prm/R2_mean"] == pytest.approx((0.8 + 0.3) / 2)
        assert m["prm/R3_mean"] == pytest.approx((0.6 + 0.9) / 2)
        assert m["prm/R4_mean"] == pytest.approx((0.4 + 0.7) / 2)

    def test_partial_dim_scores(self):
        """Rubrics absent from some samples are still averaged over those that have them."""
        scores_a = {"R1": 0.9, "R2": 0.8}          # no R3/R4
        scores_b = {"R1": 0.7, "R2": 0.6, "R3": 0.5, "R4": 0.3}
        samples = [
            _make_sample(_success(scores=scores_a)),
            _make_sample(_success(scores=scores_b)),
        ]
        m = _compute_prm_metrics(samples)
        assert m["prm/R1_mean"] == pytest.approx((0.9 + 0.7) / 2)
        assert m["prm/R3_mean"] == pytest.approx(0.5)
        assert "prm/R4_mean" in m

    def test_no_dim_scores_in_failed_calls(self):
        """Timeout samples have empty scores; rubrics must still aggregate from successes only."""
        samples = [
            _make_sample(_success(scores={"R1": 0.8, "R2": 0.6, "R3": 0.4, "R4": 0.2})),
            _make_sample(_timeout()),  # scores={}
        ]
        m = _compute_prm_metrics(samples)
        assert m["prm/R1_mean"] == pytest.approx(0.8)
