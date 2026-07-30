import math

import pytest
import torch

from slime.utils.ppo_utils import _build_per_step_discount_from_mask


def _expected_from_mask(mask: list[int], reward: float, gamma: float) -> list[float]:
    """Reference implementation for the per-step discounted return."""
    # Identify contiguous runs of 1s.
    steps: list[tuple[int, int]] = []  # (start, end_exclusive)
    i = 0
    while i < len(mask):
        if mask[i] == 1:
            j = i
            while j < len(mask) and mask[j] == 1:
                j += 1
            steps.append((i, j))
            i = j
        else:
            i += 1

    out = [0.0] * len(mask)
    n = len(steps)
    for k, (s, e) in enumerate(steps):
        discount = gamma ** (n - 1 - k)
        for t in range(s, e):
            out[t] = reward * discount
    return out


def test_no_steps_returns_zero():
    mask = torch.zeros(8, dtype=torch.int64)
    out = _build_per_step_discount_from_mask(mask, reward=1.0, gamma=0.9)
    assert torch.all(out == 0.0)


def test_single_step_full_reward():
    # one contiguous run -> no discount
    mask = torch.tensor([0, 1, 1, 1, 0, 0], dtype=torch.int64)
    out = _build_per_step_discount_from_mask(mask, reward=2.0, gamma=0.5)
    expected = torch.tensor([0, 2, 2, 2, 0, 0], dtype=torch.float32)
    assert torch.allclose(out, expected)


def test_multi_step_discount():
    # 3 steps -> discounts gamma^2, gamma^1, gamma^0
    mask = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1, 1, 1], dtype=torch.int64)
    reward = 1.0
    gamma = 0.5
    out = _build_per_step_discount_from_mask(mask, reward=reward, gamma=gamma)
    expected = torch.tensor(
        [0, 0.25, 0.25, 0, 0.5, 0, 0, 1.0, 1.0, 1.0],
        dtype=torch.float32,
    )
    assert torch.allclose(out, expected), f"got {out.tolist()} expected {expected.tolist()}"


def test_starts_with_one():
    # step at index 0
    mask = torch.tensor([1, 1, 0, 1], dtype=torch.int64)
    out = _build_per_step_discount_from_mask(mask, reward=3.0, gamma=0.5)
    expected = torch.tensor([3.0 * 0.5, 3.0 * 0.5, 0.0, 3.0], dtype=torch.float32)
    assert torch.allclose(out, expected)


def test_gamma_one_broadcasts_reward():
    # gamma=1 -> all assistant tokens get full reward, observation tokens 0
    mask = torch.tensor([1, 0, 1, 0, 1], dtype=torch.int64)
    out = _build_per_step_discount_from_mask(mask, reward=4.0, gamma=1.0)
    expected = torch.tensor([4.0, 0.0, 4.0, 0.0, 4.0], dtype=torch.float32)
    assert torch.allclose(out, expected)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_random_matches_reference(seed):
    g = torch.Generator().manual_seed(seed)
    mask = (torch.rand(64, generator=g) > 0.4).to(torch.int64)
    reward = float(torch.randn(1, generator=g).item())
    gamma = 0.7
    out = _build_per_step_discount_from_mask(mask, reward=reward, gamma=gamma)
    expected = torch.tensor(
        _expected_from_mask(mask.tolist(), reward, gamma), dtype=torch.float32
    )
    assert torch.allclose(out, expected, atol=1e-6), (
        f"mismatch\nmask={mask.tolist()}\nout={out.tolist()}\nexp={expected.tolist()}"
    )


def test_negative_reward():
    mask = torch.tensor([1, 1, 0, 1], dtype=torch.int64)
    out = _build_per_step_discount_from_mask(mask, reward=-2.0, gamma=0.5)
    # 2 steps: first gets gamma^1 * -2 = -1, second gets -2
    expected = torch.tensor([-1.0, -1.0, 0.0, -2.0], dtype=torch.float32)
    assert torch.allclose(out, expected)


def test_dtype_and_shape():
    mask = torch.tensor([0, 1, 1, 0, 1], dtype=torch.int64)
    out = _build_per_step_discount_from_mask(mask, reward=1.0, gamma=0.9)
    assert out.dtype == torch.float32
    assert out.shape == mask.shape


def test_geometric_sum_property():
    # If each step has length 1 and reward = 1 with gamma=g, the sum across
    # tokens equals the geometric sum 1 + g + g^2 + ... + g^(N-1).
    n = 5
    mask = torch.tensor([1, 0] * n, dtype=torch.int64)[:-1]  # 1 0 1 0 1 0 1 0 1
    out = _build_per_step_discount_from_mask(mask, reward=1.0, gamma=0.8)
    expected_sum = sum(0.8**k for k in range(n))
    assert math.isclose(out.sum().item(), expected_sum, abs_tol=1e-6)
