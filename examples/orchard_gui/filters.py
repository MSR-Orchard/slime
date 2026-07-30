"""Dynamic sampling filters for the multi-turn browser rollout.

``generate_browser.generate_turn_sample`` returns **one Sample per turn**, so each
element of a rollout group is a ``list[Sample]`` (a full trajectory), not a single
``Sample``. slime's stock ``check_reward_nonzero_std`` assumes flat samples and
crashes with ``'list' object has no attribute 'get_reward_value'``.

This variant collapses each trajectory to a single reward (all turns of a
trajectory share the same ``sample.reward``) before checking whether the group's
rewards have non-zero std — i.e. whether the group carries any GRPO learning
signal. Aborted/unscored trajectories (reward ``None``) count as 0.0.
"""

import torch

from slime.rollout.filter_hub.base_types import DynamicFilterOutput

__all__ = ["check_reward_nonzero_std", "check_reward_nonempty_nonzero_std"]


def _trajectory_reward(item, args) -> float:
    """One reward per group element, whether it's a Sample or a trajectory list.

    For a trajectory we use the last turn (it carries ``is_last_turn`` and the
    final reward); every turn shares the trajectory reward anyway.
    """
    sample = item[-1] if isinstance(item, list) else item
    reward = sample.get_reward_value(args)
    return float(reward) if reward is not None else 0.0


def check_reward_nonzero_std(args, samples, **kwargs):
    rewards = [_trajectory_reward(item, args) for item in samples]
    keep = torch.tensor(rewards, dtype=torch.float64).std() > 1e-6
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(rewards[0], 1)}",
    )


def _trainable_trajectory_reward(item, args) -> float | None:
    """Reward of a group element, or ``None`` when it is not trainable.

    ``reward_func`` marks judge-failed / aborted trajectories ``remove_sample=True``
    with a placeholder reward of 0.0 (propagated to every turn). That 0.0 is *not* a
    real score, so such trajectories are excluded from the group's learning-signal
    check (return ``None``) instead of being counted as a genuine 0.0 — otherwise the
    placeholder fabricates variance and keeps a group that carries no real signal.
    """
    sample = item[-1] if isinstance(item, list) else item
    if sample.remove_sample:
        return None
    reward = sample.get_reward_value(args)
    return float(reward) if reward is not None else None


def check_reward_nonempty_nonzero_std(args, samples, **kwargs):
    """Whole-group keep/drop that ignores non-trainable (judge-failed / aborted) trajectories.

    Keep the prompt-group only if **at least two trainable** trajectories carry a
    reward and those rewards have non-zero std — i.e. the group still gives GRPO a
    learning signal once ``remove_sample`` trajectories are excluded. Those turns are
    dropped from the loss separately (``remove_sample`` zeroes their mask), so this
    filter only decides whether the *whole* prompt-group is worth keeping; it never
    changes group size (no subset is returned).
    """
    rewards = [r for r in (_trainable_trajectory_reward(item, args) for item in samples) if r is not None]
    if len(rewards) < 2:
        return DynamicFilterOutput(keep=False, reason=f"insufficient_trainable_rewards_{len(rewards)}")
    keep = torch.tensor(rewards, dtype=torch.float64).std() > 1e-6
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(rewards[0], 1)}",
    )
