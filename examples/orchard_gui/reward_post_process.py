"""Per-trajectory GRPO reward normalization for the multi-turn browser rollout.

Wired via ``--custom-reward-post-process-path`` (slime calls it from
``RolloutController._post_process_rewards``).

``generate_turn_sample`` emits **one Sample per turn**, and ``reward_func`` gives every
turn of a trajectory the same reward (and flags judge-failed / aborted trajectories
``remove_sample=True`` with a placeholder reward of 0.0). slime's default
``_post_process_rewards`` reshapes the flat reward list by ``n_samples_per_prompt``,
which is wrong here: the flat list has one entry **per turn** (variable per trajectory),
not one per trajectory, so the reshape collapses to a single global row and centers
rewards across *all* prompts instead of within each prompt-group.

This hook instead:
  1. groups samples by prompt-group (``group_index``) then trajectory (``rollout_id``);
  2. reduces each trajectory to one reward (all its turns share it);
  3. centers (and optionally std-normalizes) over the **trainable** trajectories of each
     prompt-group — ``remove_sample`` trajectories are excluded from the baseline because
     their 0.0 is a placeholder, not a real score;
  4. broadcasts each trajectory's normalized advantage to all of its turns.

``remove_sample`` turns get advantage 0.0; they are masked out of the loss anyway. The
return contract matches the default: ``(raw_rewards, advantages)``, both per-sample.
"""

import torch

from slime.utils.misc import group_by
from slime.utils.types import Sample


def _trajectory_reward_and_trainable(turn_samples: list[Sample], args) -> tuple[float, bool]:
    """One reward per trajectory (every turn shares it) and whether it is trainable.

    Non-trainable == ``reward_func`` set ``remove_sample=True`` on the trajectory
    (judge failure / abort). Its 0.0 reward is a placeholder and must be kept out of
    the GRPO baseline so it does not bias the other trajectories' advantages.
    """
    reward = turn_samples[0].get_reward_value(args)
    reward = float(reward) if reward is not None else 0.0
    trainable = any(not s.remove_sample for s in turn_samples)
    return reward, trainable


def grpo_normalize_per_trajectory(args, samples: list[Sample]):
    """Center/normalize rewards per prompt-group at trajectory granularity.

    Returns ``(raw_rewards, advantages)`` — both per-sample (one entry per turn) —
    so it is a drop-in for slime's default ``_post_process_rewards``.
    """
    raw_rewards = [s.get_reward_value(args) for s in samples]

    estimator = getattr(args, "advantage_estimator", None)
    if not (
        estimator in ("grpo", "gspo", "reinforce_plus_plus_baseline")
        and getattr(args, "rewards_normalization", True)
    ):
        return raw_rewards, list(raw_rewards)

    std_norm = estimator in ("grpo", "gspo") and getattr(args, "grpo_std_normalization", True)

    advantages = [0.0] * len(samples)
    indexed = list(enumerate(samples))
    # prompt-group (group_index) -> trajectory (rollout_id) -> [(pos, sample), ...]
    for group_pairs in group_by(indexed, key=lambda pair: pair[1].group_index).values():
        trajectories = group_by(group_pairs, key=lambda pair: pair[1].rollout_id)

        traj_reward: dict = {}
        traj_trainable: dict = {}
        for tid, pairs in trajectories.items():
            traj_reward[tid], traj_trainable[tid] = _trajectory_reward_and_trainable(
                [s for _, s in pairs], args
            )

        baseline = [traj_reward[tid] for tid in trajectories if traj_trainable[tid]]
        if not baseline:
            # whole prompt-group is non-trainable; every turn is masked, leave 0.0
            continue
        baseline_t = torch.tensor(baseline, dtype=torch.float)
        mean = baseline_t.mean().item()
        # torch.std is unbiased (ddof=1), matching slime's default group-norm.
        denom = (baseline_t.std().item() + 1e-6) if (std_norm and baseline_t.numel() > 1) else 1.0

        for tid, pairs in trajectories.items():
            if not traj_trainable[tid]:
                continue  # masked out of the loss; leave advantage 0.0
            adv = (traj_reward[tid] - mean) / denom
            for pos, _ in pairs:
                advantages[pos] = adv

    return raw_rewards, advantages
