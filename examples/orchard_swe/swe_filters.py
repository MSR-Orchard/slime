"""Custom rollout filters for SWE training.

Two independent, composable hooks:

1. ``keep_group_by_pass_rate`` -- a *dynamic sampling filter*
   (``--dynamic-sampling-filter-path``). Runs once per group right after the
   group's rewards are computed and decides whether to keep the whole group.
   Keep probability is ``min(1, 1 - pass_rate)`` where ``pass_rate`` is the
   in-group realtime pass rate (#positive-reward samples / group size). Easy
   groups (high pass rate) are dropped more often; hard groups are kept.

2. ``keep_only_positive_samples`` -- a *rollout sample filter*
   (``--rollout-sample-filter-path``). Runs once on the assembled batch and
   masks out every non-positive sample by setting ``remove_sample=True`` so
   only positive-reward rollouts contribute to the training loss. The masked
   (negative) samples still remain in the group, so they keep providing the
   GRPO advantage baseline; they simply receive zero gradient.
"""

import random

from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.types import Sample


def _flatten(samples) -> list[Sample]:
    """Flatten a (possibly nested) group into a flat list of Sample objects."""
    flat = []
    for s in samples:
        if isinstance(s, list):
            flat.extend(_flatten(s))
        else:
            flat.append(s)
    return flat


def _is_positive(args, sample: Sample) -> bool:
    if sample.status == Sample.Status.ABORTED:
        return False
    reward = sample.get_reward_value(args)
    return reward is not None and reward > 0


def keep_group_by_pass_rate(args, samples: list[Sample], **kwargs) -> DynamicFilterOutput:
    """Group-level dynamic sampling filter.

    Keep the group with probability ``min(1, 1 - pass_rate)``. Groups that
    contain any ABORTED sample, or that have no positive-reward sample (nothing
    to train on once the positive-only mask is applied), are always dropped.
    """
    group = _flatten(samples)

    # Never train on a crashed group: an ABORTED sample (empty tokens / None
    # reward) would poison the training step.
    if any(s.status == Sample.Status.ABORTED for s in group):
        return DynamicFilterOutput(keep=False, reason="aborted")

    num_positive = sum(1 for s in group if _is_positive(args, s))

    # If no positive sample survives, the positive-only mask would zero out the
    # whole group -> no gradient. Drop it and let the loop fetch more data.
    if num_positive == 0:
        return DynamicFilterOutput(keep=False, reason="no_positive")

    pass_rate = num_positive / len(group)
    keep_prob = min(1.0, 1.0 - pass_rate)

    keep = random.random() < keep_prob
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"pass_rate_drop_{round(pass_rate, 3)}",
    )


def keep_only_positive_samples(args, groups: list[list[Sample]]) -> None:
    """Per-sample rollout filter: keep only positive-reward rollouts.

    Marks every non-positive sample with ``remove_sample=True`` (its loss mask
    is zeroed downstream) so only positive-reward rollouts produce gradient.
    """
    for group in groups:
        for sample in _flatten(group):
            sample.remove_sample = not _is_positive(args, sample)
