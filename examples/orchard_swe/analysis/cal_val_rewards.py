import json
import argparse
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
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

# read jsonl file and get all instance_id into a set
def get_instance_id_set(jsonl_file):
    instance_id_set = set()
    with open(jsonl_file, "r") as f:
        for line in f:
            instance = json.loads(line)
            instance_id = instance["metadata"].get("instance_id", None)
            if instance_id is not None:
                instance_id_set.add(instance_id)
    return instance_id_set


# go through all files and get the reward for each instance_id, then calculate the average reward for all instance_ids
def calculate_average_reward(json_files, reward_files_set):
    """Compute rewards for the given json files in parallel.

    `reward_files_set` is the set of existing `*_rewards.json` paths in the
    folder, so we avoid an extra `Path.exists()` stat per file.
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

    # Coverage diagnostics: distinguish a real low score from one caused by
    # missing/failed reward files (which are silently counted as 0.0 below).
    missing_reward_count = 0  # trajectory present but no *_rewards.json (scored 0)
    unreadable_count = 0      # trajectory json could not be read (dropped entirely)
    for (instance_id, _path, is_reward_file), instance in zip(tasks, results):
        if instance is None:
            unreadable_count += 1
            continue
        if is_reward_file:
            reward = 1.0 if instance.get("resolved", False) else 0.0
        else:
            instance_id = instance["instance_id"]
            reward = 0.0
            missing_reward_count += 1
        instance_rewards.setdefault(instance_id, []).append(reward)

    average_rewards = {instance_id: sum(rewards) / len(rewards) for instance_id, rewards in instance_rewards.items()}
    # calculate the overall average reward across all rewards
    all_rewards = []

    for instance_id, rewards in instance_rewards.items():
        all_rewards.extend(rewards)

    micro_average_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0

    macro_average_reward = sum(average_rewards.values()) / len(average_rewards) if average_rewards else 0.0
    coverage = {
        "total_trajectories": len(json_files),
        "missing_reward_files": missing_reward_count,
        "unreadable_trajectories": unreadable_count,
    }
    return average_rewards, micro_average_reward, macro_average_reward, coverage

# go through all files in a folder and keep all files paths that contain a instance_id in the instance_id_set
def get_json_files_with_instance_ids(folder_path, instance_id_set):
    """Scan the folder once with os.scandir, returning grouped json files,
    the rl-step list, and the set of existing `*_rewards.json` paths.
    """
    index2file = {}
    reward_files_set = set()
    with os.scandir(folder_path) as it:
        for entry in it:
            name = entry.name
            if not entry.is_file() or not name.endswith(".json"):
                continue
            stem = name[:-5]  # strip .json
            if stem.endswith("_rewards"):
                reward_files_set.add(entry.path)
                continue
            # example: astropy__astropy-13977_4729
            try:
                idx_str = stem.rsplit("_", 1)[1]
                index = int(idx_str)
            except (IndexError, ValueError):
                continue
            fields = stem.split("_")[:-1]
            instance_id = "_".join(fields)
            if instance_id in instance_id_set:
                index2file[index] = entry.path

    if not index2file:
        return [], [], reward_files_set

    sorted_indexes = sorted(index2file.keys())
    file_groups = [[]]
    rl_step_list = []
    previous_index = sorted_indexes[0]
    for index in sorted_indexes:
        if not rl_step_list:
            rl_step_list.append(index // 100000 + 1)
        if index - previous_index > 100:
            file_groups.append([index2file[previous_index]])
            rl_step_list.append(index // 100000 + 1)
        else:
            file_groups[-1].append(index2file[index])
        previous_index = index
    assert len(file_groups) == len(rl_step_list)
    return file_groups, rl_step_list, reward_files_set


def get_json_files_by_step_folders(folder_path, instance_id_set):
    """New layout: trajectories live in per-RL-step subfolders named
    ``rl_step_<N>`` (created by swe_generate_v2.py). Each subfolder contains
    the trajectory files (``<instance>_<index>.json``) and their reward files
    (``<instance>_<index>_rewards.json``).

    Returns the same triple as :func:`get_json_files_with_instance_ids`:
    ``(file_groups, rl_step_list, reward_files_set)`` where group ``i``
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
                stem = name[:-5]  # strip .json
                if stem.endswith("_rewards"):
                    reward_files_set.add(entry.path)
                    continue
                fields = stem.split("_")[:-1]
                instance_id = "_".join(fields)
                if instance_id in instance_id_set:
                    group.append(entry.path)
        if group:
            file_groups.append(group)
            rl_step_list.append(step)

    assert len(file_groups) == len(rl_step_list)
    return file_groups, rl_step_list, reward_files_set


def copy_files_into_step_folders(json_file_groups, rl_step_list, reward_files_set, dest_folder):
    """Copy each group's trajectory + reward files into a per-RL-step subfolder.

    For group ``i`` (RL step ``rl_step_list[i]``), files are copied into
    ``dest_folder/step_{rl_step}_val/``. Both the trajectory file
    (``<instance>_<index>.json``) and its reward file
    (``<instance>_<index>_rewards.json``, when present) are copied.
    """
    copy_tasks = []  # list of (src, dst)
    for group, rl_step in zip(json_file_groups, rl_step_list):
        step_dir = Path(dest_folder) / f"step_{rl_step:02d}_val"
        step_dir.mkdir(parents=True, exist_ok=True)
        for json_file in group:
            json_file_str = str(json_file)
            copy_tasks.append((json_file_str, str(step_dir / Path(json_file_str).name)))
            reward_file = json_file_str.replace(".json", "_rewards.json")
            if reward_file in reward_files_set:
                copy_tasks.append((reward_file, str(step_dir / Path(reward_file).name)))

    def _copy(task):
        src, dst = task
        try:
            shutil.copy2(src, dst)
        except OSError as e:
            print(f"Warning: Failed to copy {src} -> {dst}: {e}")

    with ThreadPoolExecutor(max_workers=_NUM_WORKERS) as executor:
        list(executor.map(_copy, copy_tasks))
    print(f"Copied {len(copy_tasks)} files into per-step subfolders under {dest_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate average rewards for SWE instances")
    parser.add_argument("--instance-jsonl", type=str, nargs="+", required=True, help="Path(s) to the JSONL file(s) containing instance metadata")
    parser.add_argument("--reward-folder", type=str, required=True, help="Path to the folder containing reward JSONL files")
    parser.add_argument("--visual-folder", type=str, default="/data/users/xxxx/workspace/swe_trajectories/figures", help="Path to the folder to save the reward figure")
    parser.add_argument("--copy-step-folders", action="store_true", help="Copy trajectory + reward files into per-RL-step subfolders (step_xx_val/)")
    parser.add_argument("--dest-folder", type=str, default=None, help="Destination folder for the per-step subfolders (defaults to --reward-folder)")
    parser.add_argument("--rl-step-subfolders", action="store_true", help="Read trajectories from per-RL-step subfolders named rl_step_<N> (new layout). Default reads all trajectories from a single flat folder (old layout).")
    args = parser.parse_args()

    instance_id_set = set()
    for jsonl_file in args.instance_jsonl:
        instance_id_set |= get_instance_id_set(jsonl_file)
    print(list(instance_id_set)[:3])
    print(f"Found {len(instance_id_set)} unique instance IDs from {len(args.instance_jsonl)} file(s)")

    json_file_groups, rl_step_list, reward_files_set = (
        get_json_files_by_step_folders(args.reward_folder, instance_id_set)
        if args.rl_step_subfolders
        else get_json_files_with_instance_ids(args.reward_folder, instance_id_set)
    )
    for group in json_file_groups:
        print(group[:3])
    print(f"Found {len(json_file_groups)} groups of reward files in {args.reward_folder} that match instance IDs")
    print(f"RL step list: {rl_step_list}")

    if args.copy_step_folders:
        dest_folder = args.dest_folder or args.reward_folder
        copy_files_into_step_folders(json_file_groups, rl_step_list, reward_files_set, dest_folder)

    macro_rewards_list = []
    for i, group in enumerate(json_file_groups):
        average_rewards, micro_average_reward, macro_average_reward, coverage = calculate_average_reward(group, reward_files_set)
        macro_rewards_list.append(macro_average_reward)
        total = coverage["total_trajectories"]
        missing = coverage["missing_reward_files"]
        unreadable = coverage["unreadable_trajectories"]
        missing_pct = (100.0 * missing / total) if total else 0.0
        print(f"Average rewards in RL step {rl_step_list[i]}: [Micro] {micro_average_reward:.4f}; [Macro] {macro_average_reward:.4f}; number of instances: {len(average_rewards)}")
        print(f"  Coverage: {total} trajectories, {missing} missing reward files counted as 0 ({missing_pct:.1f}%), {unreadable} unreadable")
        # print("Average rewards for each instance ID:")
        # for instance_id, avg_reward in average_rewards.items():
        #     print(f"Instance ID: {instance_id}, Average Reward: {avg_reward:.4f}")

    # Plot macro rewards across RL steps
    visual_folder = Path(args.visual_folder)
    visual_folder.mkdir(parents=True, exist_ok=True)
    fig_name = Path(args.reward_folder).name + ".pdf"

    # # !!!!!!!!!! temporary hack to add the reward for the first RL step
    # if rl_step_list[0] != 1:
    #     rl_step_list.insert(0, 1)
    #     macro_rewards_list.insert(0, 0.5)

    #success_rate_list = [reward / 10 * 100 for reward in macro_rewards_list] # in percentage
    success_rate_list = [reward * 100 for reward in macro_rewards_list] # in percentage

    plt.figure(figsize=(10, 6))
    #plt.plot(rl_step_list, macro_rewards_list, marker='o', linewidth=2, markersize=6)
    plt.plot(rl_step_list, success_rate_list, marker='o', linewidth=2, markersize=6)
    plt.xlabel("RL Steps", fontsize=14)
    #plt.ylabel("Macro Reward", fontsize=14)
    plt.ylabel("Success Rate (%)", fontsize=14)
    #plt.title("Macro Reward across RL Steps", fontsize=16)
    plt.title("Success Rate across RL Steps", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    save_path = visual_folder / fig_name
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Figure saved to {save_path}")