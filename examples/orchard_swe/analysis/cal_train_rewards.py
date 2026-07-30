"""Calculate average training rewards across RL steps.

This is the training-time counterpart of ``cal_val_rewards.py``. Instead of
keeping only the trajectories whose ``instance_id`` is in a validation set, it
keeps every *training* trajectory and excludes the validation ones, then reports
the average reward per RL step.

Training trajectories are written by ``swe_generate_v2.py`` with file names
``<instance_id>_<index>.json`` (and reward files ``<instance_id>_<index>_rewards.json``).
Validation instance IDs are read from one or more pre-defined JSONL files (the
same files used to evaluate the model); any trajectory whose ``instance_id`` is
in that set is excluded.

Two grouping modes are supported:

1. ``step-folders`` (default): trajectories live in per-RL-step subfolders named
   ``rl_step_<N>`` (the new layout produced by ``swe_generate_v2.py``). Each
   subfolder is treated as one RL step; validation trajectories inside it are
   excluded.

2. ``index-window``: trajectories live in a single flat folder. They are grouped
   into RL steps by their monotonically increasing index, ``window_size`` indices
   per step (default 128). Validation trajectories are excluded.
"""

import json
import argparse
import os
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

# Number of worker threads for parallel JSON reads (I/O-bound).
_NUM_WORKERS = min(64, (os.cpu_count() or 4) * 4)


def _read_json(path):
    """Read a JSON file and return its parsed contents, or None on failure."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: Failed to read {path}: {e}. Skipping.")
        return None


def get_instance_id_set(jsonl_file):
    """Read a JSONL file and collect every ``instance_id`` into a set."""
    instance_id_set = set()
    with open(jsonl_file, "r") as f:
        for line in f:
            instance = json.loads(line)
            instance_id = instance["metadata"].get("instance_id", None)
            if instance_id is not None:
                instance_id_set.add(instance_id)
    return instance_id_set


def calculate_average_reward(json_files, reward_files_set):
    """Compute rewards for the given json files in parallel.

    ``reward_files_set`` is the set of existing ``*_rewards.json`` paths, so we
    avoid an extra ``Path.exists()`` stat per file. A trajectory with a reward
    file gets reward ``1.0`` when resolved, otherwise ``0.0``; a trajectory
    without a reward file is treated as unresolved (reward ``0.0``).
    """
    instance_rewards = {}

    # Decide for each file which path to actually read (reward file vs original).
    tasks = []  # list of (instance_id, path_to_read, is_reward_file)
    for json_file in json_files:
        json_file_str = str(json_file)
        possible_reward_file = json_file_str.replace(".json", "_rewards.json")
        if possible_reward_file in reward_files_set:
            fields = possible_reward_file.split("/")[-1].split("_rewards")[0].split("_")[:-1]
            instance_id = "_".join(fields)
            tasks.append((instance_id, possible_reward_file, True))
        else:
            tasks.append((None, json_file_str, False))

    # Read all chosen files in parallel.
    paths = [t[1] for t in tasks]
    with ThreadPoolExecutor(max_workers=_NUM_WORKERS) as executor:
        results = list(executor.map(_read_json, paths))

    for (instance_id, _path, is_reward_file), instance in zip(tasks, results):
        if instance is None:
            continue
        if is_reward_file:
            reward = 1.0 if instance["resolved"] else 0.0
        else:
            instance_id = instance["instance_id"]
            reward = 0.0
        instance_rewards.setdefault(instance_id, []).append(reward)

    average_rewards = {instance_id: sum(rewards) / len(rewards) for instance_id, rewards in instance_rewards.items()}
    # calculate the overall average reward across all rewards
    all_rewards = []
    for instance_id, rewards in instance_rewards.items():
        all_rewards.extend(rewards)

    micro_average_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
    macro_average_reward = sum(average_rewards.values()) / len(average_rewards) if average_rewards else 0.0
    return average_rewards, micro_average_reward, macro_average_reward


def _parse_trajectory_file(name):
    """Parse a trajectory file name ``<instance_id>_<index>.json``.

    Returns ``(instance_id, index)`` or ``None`` if the name is not a trajectory
    file (e.g. a ``*_rewards.json`` reward file, a ``*_group_info.jsonl`` file, or
    any other non-matching name).
    """
    if not name.endswith(".json"):
        return None
    stem = name[:-5]  # strip .json
    if stem.endswith("_rewards"):
        return None
    # example: astropy__astropy-13977_4729
    try:
        index = int(stem.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return None
    instance_id = "_".join(stem.split("_")[:-1])
    return instance_id, index


def get_json_files_by_step_folders(folder_path, val_instance_ids):
    """Option 1: per-RL-step subfolders named ``rl_step_<N>``.

    Each subfolder contains the trajectory files (``<instance>_<index>.json``)
    and their reward files (``<instance>_<index>_rewards.json``). Validation
    trajectories (instance_id in ``val_instance_ids``) are excluded.

    Returns ``(file_groups, rl_step_list, reward_files_set)`` where group ``i``
    corresponds to RL step ``rl_step_list[i]``.
    """
    step_dirs = []
    with os.scandir(folder_path) as it:
        for entry in it:
            if not entry.is_dir() or not entry.name.startswith("rl_step_"):
                continue
            try:
                step = int(entry.name[len("rl_step_"):])
            except ValueError:
                continue
            step_dirs.append((step, entry.path))

    step_dirs.sort(key=lambda x: x[0])

    file_groups = []
    rl_step_list = []
    reward_files_set = set()
    for step, dir_path in step_dirs:
        group = []
        with os.scandir(dir_path) as it:
            for entry in it:
                name = entry.name
                if not entry.is_file() or not name.endswith(".json"):
                    continue
                if name[:-5].endswith("_rewards"):
                    reward_files_set.add(entry.path)
                    continue
                parsed = _parse_trajectory_file(name)
                if parsed is None:
                    continue
                instance_id, index = parsed
                if instance_id in val_instance_ids:
                    continue
                group.append(entry.path)
        if group:
            file_groups.append(group)
            rl_step_list.append(step)

    assert len(file_groups) == len(rl_step_list)
    return file_groups, rl_step_list, reward_files_set


def get_json_files_by_index_window(folder_path, window_size, val_instance_ids):
    """Option 2: single flat folder grouped by monotonically increasing index.

    Training trajectories are grouped into RL steps by ``index // window_size``
    (``window_size`` indices per step, default 128). Validation trajectories
    (instance_id in ``val_instance_ids``) are excluded.

    Returns ``(file_groups, rl_step_list, reward_files_set)`` where group ``i``
    corresponds to RL step ``rl_step_list[i]``.
    """
    step2files = {}
    reward_files_set = set()
    with os.scandir(folder_path) as it:
        for entry in it:
            name = entry.name
            if not entry.is_file() or not name.endswith(".json"):
                continue
            if name[:-5].endswith("_rewards"):
                reward_files_set.add(entry.path)
                continue
            parsed = _parse_trajectory_file(name)
            if parsed is None:
                continue
            instance_id, index = parsed
            if instance_id in val_instance_ids:
                continue
            step = index // window_size + 1
            step2files.setdefault(step, []).append(entry.path)

    rl_step_list = sorted(step2files.keys())
    file_groups = [step2files[step] for step in rl_step_list]
    assert len(file_groups) == len(rl_step_list)
    return file_groups, rl_step_list, reward_files_set


def _moving_average(values, window):
    """Trailing moving average over ``values`` using a window of ``window``.

    Each point ``i`` averages ``values[max(0, i - window + 1) : i + 1]`` so the
    output has the same length as the input (no shortening at the start).
    """
    window = max(1, window)
    averaged = []
    for i in range(len(values)):
        chunk = values[max(0, i - window + 1) : i + 1]
        averaged.append(sum(chunk) / len(chunk))
    return averaged


def plot_rewards(rl_step_list, micro_rewards_list, macro_rewards_list, reward_folder, ma_window):
    """Plot micro/macro success rate across RL steps with a moving-average line,
    saving the figure into ``reward_folder``."""
    micro_rate = [r * 100 for r in micro_rewards_list]  # in percentage
    macro_rate = [r * 100 for r in macro_rewards_list]  # in percentage
    macro_ma = _moving_average(macro_rate, ma_window)

    plt.figure(figsize=(12, 6))
    plt.plot(rl_step_list, micro_rate, marker="o", linewidth=1, markersize=4, alpha=0.4, color="tab:orange", label="Micro success rate")
    plt.plot(rl_step_list, macro_rate, marker="o", linewidth=1, markersize=4, alpha=0.4, color="tab:blue", label="Macro success rate")
    plt.plot(rl_step_list, macro_ma, linewidth=2.5, color="tab:red", label=f"Macro moving avg (window={ma_window})")
    plt.xlabel("RL Steps", fontsize=14)
    plt.ylabel("Success Rate (%)", fontsize=14)
    plt.title("Training Success Rate across RL Steps", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()

    fig_name = os.path.basename(os.path.normpath(reward_folder)) + "_train_rewards.pdf"
    save_path = os.path.join(reward_folder, fig_name)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Figure saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate average training rewards across RL steps")
    parser.add_argument("--reward-folder", type=str, required=True, help="Path to the folder containing training trajectory / reward files")
    parser.add_argument(
        "--val-instance-jsonl",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to the JSONL file(s) containing validation instance metadata; trajectories with these instance_ids are excluded",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["step-folders", "index-window"],
        default="step-folders",
        help="step-folders: one RL step per rl_step_<N> subfolder (new layout). "
        "index-window: single flat folder grouped by every --window-size mono-increasing index.",
    )
    parser.add_argument("--window-size", type=int, default=128, help="Number of mono-increasing indices per RL step (index-window mode)")
    parser.add_argument("--ma-window", type=int, default=5, help="Window size (number of RL steps) for the moving-average line in the plot")
    args = parser.parse_args()

    val_instance_ids = set()
    for jsonl_file in args.val_instance_jsonl:
        val_instance_ids |= get_instance_id_set(jsonl_file)
    print(f"Excluding {len(val_instance_ids)} validation instance IDs from {len(args.val_instance_jsonl)} file(s)")

    if args.mode == "step-folders":
        json_file_groups, rl_step_list, reward_files_set = get_json_files_by_step_folders(
            args.reward_folder, val_instance_ids
        )
    else:
        json_file_groups, rl_step_list, reward_files_set = get_json_files_by_index_window(
            args.reward_folder, args.window_size, val_instance_ids
        )

    for group in json_file_groups:
        print(group[:3])
    print(f"Found {len(json_file_groups)} RL-step group(s) of training trajectories in {args.reward_folder}")
    print(f"RL step list: {rl_step_list}")

    micro_rewards_list = []
    macro_rewards_list = []
    for i, group in enumerate(json_file_groups):
        average_rewards, micro_average_reward, macro_average_reward = calculate_average_reward(group, reward_files_set)
        micro_rewards_list.append(micro_average_reward)
        macro_rewards_list.append(macro_average_reward)
        print(
            f"Average rewards in RL step {rl_step_list[i]}: "
            f"[Micro] {micro_average_reward:.4f}; [Macro] {macro_average_reward:.4f}; "
            f"number of trajectories: {len(group)}; number of instances: {len(average_rewards)}"
        )

    if rl_step_list:
        plot_rewards(
            rl_step_list,
            micro_rewards_list,
            macro_rewards_list,
            args.reward_folder,
            args.ma_window,
        )
    else:
        print("No training trajectories found; nothing to plot.")
