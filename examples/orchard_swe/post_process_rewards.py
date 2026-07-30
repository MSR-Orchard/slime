import logging

from slime.utils.types import Sample

logger = logging.getLogger(__name__)


def naive_reinforce(args, samples: list[Sample], **kwargs):
    """Return raw rewards without any normalization. Replaces None rewards with 0.0."""
    raw_rewards = [sample.get_reward_value(args) for sample in samples]

    none_count = sum(1 for r in raw_rewards if r is None)
    if none_count > 0:
        logger.warning(
            f"Found {none_count} samples with None rewards (e.g., aborted/failed samples). "
            "Replacing with 0.0."
        )
        raw_rewards = [r if r is not None else 0.0 for r in raw_rewards]

    return raw_rewards, raw_rewards
