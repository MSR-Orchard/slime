import torch

from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.types import Sample

__all__ = [
    "check_reward_nonzero_std",
    "check_no_aborted",
    "check_no_truncated",
    "check_no_aborted_and_reward_nonzero_std",
    "check_no_aborted_nonzero_std_and_pos_reward",
    "check_no_aborted_truncated_nonzero_std_and_pos_reward",
    "reject_all",
]


def check_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    rewards = [sample.get_reward_value(args) for sample in samples]
    keep = torch.tensor(rewards, dtype=torch.float64).std() > 1e-6
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(rewards[0], 1)}",
    )


def check_no_aborted(args, samples: list[Sample], **kwargs):
    """Reject groups that contain any ABORTED sample.

    ABORTED samples typically have empty tokens / zero response_length / None
    reward, which would crash or hang the training step.  Dropping the whole
    group is safe because the dynamic-sampling loop will simply fetch more data
    to fill the rollout batch.
    """
    has_aborted = any(s.status == Sample.Status.ABORTED for s in samples)
    return DynamicFilterOutput(
        keep=not has_aborted,
        reason=None if not has_aborted else "aborted",
    )


def check_no_truncated(args, samples: list[Sample], **kwargs):
    """Reject groups that contain any TRUNCATED sample.

    TRUNCATED samples hit a length / time / step limit before finishing, so they
    carry an incomplete trajectory.  Dropping the whole group keeps training to
    fully-finished rollouts only; the dynamic-sampling loop will fetch more data
    to refill the rollout batch.
    """
    has_truncated = any(s.status == Sample.Status.TRUNCATED for s in samples)
    return DynamicFilterOutput(
        keep=not has_truncated,
        reason=None if not has_truncated else "truncated",
    )


def check_no_aborted_and_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    """Combined filter: reject ABORTED samples first, then check reward std."""
    result = check_no_aborted(args, samples, **kwargs)
    if not result.keep:
        return result
    return check_reward_nonzero_std(args, samples, **kwargs)


def check_no_aborted_nonzero_std_and_pos_reward(args, samples: list[Sample], **kwargs):
    """Combined filter: reject ABORTED samples, check reward std, and require at least one positive reward."""
    result = check_no_aborted(args, samples, **kwargs)
    if not result.keep:
        return result
    result = check_reward_nonzero_std(args, samples, **kwargs)
    if not result.keep:
        return result
    rewards = [sample.get_reward_value(args) for sample in samples]
    has_positive = any(r > 0 for r in rewards)
    return DynamicFilterOutput(
        keep=has_positive,
        reason=None if has_positive else f"no_positive_reward_max_{round(max(rewards), 1)}",
    )


def check_no_aborted_truncated_nonzero_std_and_pos_reward(args, samples: list[Sample], **kwargs):
    """Combined filter: reject ABORTED and TRUNCATED samples, check reward std, and require at least one positive reward."""
    result = check_no_aborted(args, samples, **kwargs)
    if not result.keep:
        return result
    result = check_no_truncated(args, samples, **kwargs)
    if not result.keep:
        return result
    result = check_reward_nonzero_std(args, samples, **kwargs)
    if not result.keep:
        return result
    rewards = [sample.get_reward_value(args) for sample in samples]
    has_positive = any(r > 0 for r in rewards)
    return DynamicFilterOutput(
        keep=has_positive,
        reason=None if has_positive else f"no_positive_reward_max_{round(max(rewards), 1)}",
    )


def reject_all(args, samples: list[Sample], **kwargs):
    """Reject all sample groups so only rollout runs without training.

    With a 5% chance, keep the sample group to allow occasional training.
    """
    import random
    #keep = random.random() < 0.05
    keep = False
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else "reject_all",
    )
