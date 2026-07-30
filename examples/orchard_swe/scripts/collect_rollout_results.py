#!/usr/bin/env python3
"""
Collect rollout results from one or more rollout folders.

Groups trajectory and reward files by instance ID, counts resolved instances,
computes pass rates, and filters instance IDs by pass-rate range.

Usage:
    python collect_rollout_results.py \
        --folders /path/to/20260408_210651_thinking_rollout-only_vkazr0 \
                  /path/to/20260408_210146_thinking_rollout-only_hlcaqq \
        --output-dir /path/to/output \
        --min-rate 0.25 --max-rate 0.75
"""

import argparse
import json
import os
import random
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def parse_instance_id(filename: str):
    """
    Parse instance ID from a filename.

    Examples:
        durandtibo_grizz_pr526_319.json -> durandtibo_grizz_pr526
        durandtibo_grizz_pr526_319_rewards.json -> durandtibo_grizz_pr526
    """
    name = filename
    if name.endswith("_rewards.json"):
        name = name[: -len("_rewards.json")]
    elif name.endswith(".json"):
        name = name[: -len(".json")]
    else:
        return None, None

    # The instance id is everything up to the last underscore segment (which is a numeric index)
    parts = name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        instance_id = parts[0]
    else:
        instance_id = name

    is_reward = filename.endswith("_rewards.json")
    return instance_id, is_reward


def collect_from_folders(folders: list[str]) -> dict:
    """
    Scan all folders and group files by instance ID.

    Returns:
        dict: {instance_id: {"trajectory_files": [...], "reward_files": [...]}}
    """
    grouped = defaultdict(lambda: {"trajectory_files": [], "reward_files": []})

    for folder in folders:
        if not os.path.isdir(folder):
            print(f"Warning: {folder} is not a directory, skipping.")
            continue

        # os.scandir is significantly faster than os.listdir + os.path.join
        # because it avoids extra stat calls and string allocations.
        with os.scandir(folder) as it:
            for entry in it:
                fname = entry.name
                if not fname.endswith(".json"):
                    continue
                instance_id, is_reward = parse_instance_id(fname)
                if instance_id is None:
                    continue

                fpath = entry.path
                if is_reward:
                    grouped[instance_id]["reward_files"].append(fpath)
                else:
                    grouped[instance_id]["trajectory_files"].append(fpath)

    return dict(grouped)


def _check_resolved(fpath: str) -> bool:
    """Read a single reward file and return whether resolved=True."""
    try:
        with open(fpath, "rb") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: failed to read {fpath}: {e}")
        return False
    return bool(data.get("resolved", False))


def build_results(grouped: dict, max_workers: int = 32) -> dict:
    """
    Build per-instance result dict.

    Reads all reward files in parallel using a thread pool (I/O bound),
    then aggregates resolved counts per instance.

    Returns:
        dict: {instance_id: {"trajectory_files": N, "reward_files": M, "resolved": K}}
    """
    # Flatten all reward files with their owning instance id.
    file_to_instance: list[tuple[str, str]] = []
    for instance_id, files in grouped.items():
        for fpath in files["reward_files"]:
            file_to_instance.append((instance_id, fpath))

    resolved_counts: dict[str, int] = defaultdict(int)

    if file_to_instance:
        # Threads are appropriate here since the work is dominated by file I/O
        # and small JSON parsing.
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_iid = {
                executor.submit(_check_resolved, fpath): iid
                for iid, fpath in file_to_instance
            }
            for future in as_completed(future_to_iid):
                if future.result():
                    resolved_counts[future_to_iid[future]] += 1

    results = {}
    for instance_id, files in grouped.items():
        results[instance_id] = {
            "trajectory_files": len(files["trajectory_files"]),
            "reward_files": len(files["reward_files"]),
            "resolved": resolved_counts.get(instance_id, 0),
        }
    return results


def save_jsonl(results: dict, output_path: str):
    """Save results as a JSONL file (one JSON object per line)."""
    with open(output_path, "w") as f:
        for instance_id, stats in sorted(results.items()):
            record = {"instance_id": instance_id, **stats}
            f.write(json.dumps(record) + "\n")
    print(f"Saved results to {output_path}")


def load_jsonl(input_path: str) -> dict:
    """Load results from a JSONL file produced by `save_jsonl`."""
    results = {}
    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            instance_id = record.pop("instance_id")
            results[instance_id] = record
    print(f"Loaded {len(results)} instance results from {input_path}")
    return results


# Tool schema embedded in every exported record's "tools" field.
BASH_TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute",
                    }
                },
                "required": ["command"],
            },
        },
    }
]


def _reward_to_trajectory_path(reward_path: str) -> str:
    """Map a `*_rewards.json` path to its sibling trajectory `*.json` path."""
    assert reward_path.endswith("_rewards.json"), reward_path
    return reward_path[: -len("_rewards.json")] + ".json"


def _load_resolved_record(reward_path: str, rate_map: dict) -> dict | None:
    """
    If the reward file at `reward_path` has resolved=True, load the paired
    trajectory file and return the export record. Otherwise return None.
    """
    try:
        with open(reward_path, "rb") as f:
            reward_data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: failed to read {reward_path}: {e}")
        return None
    if not reward_data.get("resolved", False):
        return None

    traj_path = _reward_to_trajectory_path(reward_path)
    try:
        with open(traj_path, "rb") as f:
            orig = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: failed to read trajectory {traj_path}: {e}")
        return None

    instance_id = orig.get("instance_id")
    instance_info = orig.get("instance") or {}
    rate_info = rate_map.get(instance_id, {"rate": 0.0, "orig": "0/0"})

    record = {
        "tools": BASH_TOOL_SCHEMA,
        "messages": orig.get("messages"),
        "metadata": {
            "instance_id": instance_id,
            "dataset": instance_info.get("dataset"),
            "repo": instance_info.get("repo"),
            "n_turns": orig.get("n_steps"),
            "original_path": os.path.abspath(traj_path),
            "instance_id_resolved_rate": rate_info["rate"],
            "instance_id_resolved_orig": rate_info["orig"],
        },
    }
    return record


def build_rate_map(results: dict) -> dict:
    """
    Build instance_id -> {"rate": float, "orig": "K/N"} from per-instance stats.
    """
    rate_map = {}
    for instance_id, stats in results.items():
        total = stats["trajectory_files"]
        resolved = stats["resolved"]
        rate = (resolved / total) if total > 0 else 0.0
        rate_map[instance_id] = {"rate": rate, "orig": f"{resolved}/{total}"}
    return rate_map


def export_resolved_jsonl(grouped: dict, rate_map: dict, output_path: str,
                          max_workers: int = 32) -> int:
    """
    For every reward file with resolved=True, load the paired trajectory and
    emit one JSON record per line into `output_path`. Returns the number of
    records written.
    """
    reward_paths: list[str] = []
    for files in grouped.values():
        reward_paths.extend(files["reward_files"])

    written = 0
    with open(output_path, "w") as out_f, ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_load_resolved_record, p, rate_map) for p in reward_paths]
        for future in as_completed(futures):
            record = future.result()
            if record is None:
                continue
            out_f.write(json.dumps(record) + "\n")
            written += 1
    print(f"Saved {written} resolved trajectories to {output_path}")
    return written


def compute_pass_at_k(results: dict, numerator_key: str, denominator_key: str, k: int) -> float | None:
    """
    Compute unbiased pass@k across instances using the Codex estimator:
        pass@k(instance) = 1 - C(n - c, k) / C(n, k)   if n >= k else None
    where n = total samples (denominator) and c = correct samples (numerator).

    Returns the mean pass@k over instances with n >= k, or None if no such instance.
    """
    from math import comb

    values = []
    for stats in results.values():
        n = stats[denominator_key]
        c = stats[numerator_key]
        if n < k:
            continue
        if c >= n:
            values.append(1.0)
        elif n - c < k:
            values.append(1.0)
        else:
            values.append(1.0 - comb(n - c, k) / comb(n, k))
    if not values:
        return None
    return sum(values) / len(values)


def print_rate_distribution(results: dict, numerator_key: str, denominator_key: str,
                            title: str, bin_size: float = 0.1,
                            pass_at_ks: tuple[int, ...] = ()):
    """Print histogram of a rate (numerator_key / denominator_key)."""
    import math

    rates = []
    for instance_id, stats in results.items():
        total = stats[denominator_key]
        if total > 0:
            rate = stats[numerator_key] / total
        else:
            rate = 0.0
        rates.append(rate)

    num_bins = int(math.ceil(1.0 / bin_size))
    bins = [0] * (num_bins + 1)  # extra bin for rate == 1.0
    for r in rates:
        idx = min(int(r / bin_size), num_bins - 1)
        bins[idx] += 1

    print(f"\n{title:=^60}")
    print(f"Total instances: {len(rates)}")
    print(f"{'Bin Range':<20} {'Count':<10} {'Percentage':<10}")
    print("-" * 40)
    for i in range(num_bins):
        lo = i * bin_size
        hi = lo + bin_size
        label = f"[{lo:.2f}, {hi:.2f})"
        if i == num_bins - 1:
            label = f"[{lo:.2f}, {hi:.2f}]"
        count = bins[i]
        pct = count / len(rates) * 100 if rates else 0
        print(f"{label:<20} {count:<10} {pct:.1f}%")
    print("=" * 60)

    # Show exact 0.0 and 1.0 counts separately
    if rates:
        count_zero = sum(1 for r in rates if r == 0.0)
        count_one = sum(1 for r in rates if r == 1.0)
        n = len(rates)
        print(f"Rate = 0.0: {count_zero} ({count_zero / n * 100:.1f}%)")
        print(f"Rate = 1.0: {count_one} ({count_one / n * 100:.1f}%)")

    # Summary stats
    if rates:
        avg_rate = sum(rates) / len(rates)
        print(f"Average rate: {avg_rate:.4f}")
        for k in pass_at_ks:
            pk = compute_pass_at_k(results, numerator_key, denominator_key, k)
            n_eligible = sum(1 for s in results.values() if s[denominator_key] >= k)
            if pk is None:
                print(f"pass@{k}: N/A (no instances with >= {k} samples)")
            else:
                print(f"pass@{k}: {pk:.4f} (over {n_eligible} instances with >= {k} samples)")
        print(f"Instances with rate > 0: {sum(1 for r in rates if r > 0)}")
        print(f"Instances with rate = 1: {sum(1 for r in rates if r == 1.0)}")


def filter_instances_by_rate(results: dict, min_rate: float, max_rate: float) -> list[str]:
    """Return instance IDs whose pass rate is within [min_rate, max_rate]."""
    filtered = []
    for instance_id, stats in sorted(results.items()):
        total = stats["trajectory_files"]
        if total > 0:
            rate = stats["resolved"] / total
        else:
            rate = 0.0
        if min_rate <= rate <= max_rate:
            filtered.append(instance_id)
    return filtered


def get_zero_accuracy_instances(results: dict) -> list[str]:
    """Return instance IDs whose pass rate is exactly 0.0."""
    zero_instances = []
    for instance_id, stats in sorted(results.items()):
        total = stats["trajectory_files"]
        if total > 0:
            rate = stats["resolved"] / total
        else:
            rate = 0.0
        if rate == 0.0:
            zero_instances.append(instance_id)
    return zero_instances


def make_output_name(folders: list[str]) -> str:
    """Derive output filename suffix from folder names."""
    suffixes = []
    for folder in folders:
        name = Path(folder).name
        # Extract the short suffix (e.g., "vkazr0" from "20260408_210651_thinking_rollout-only_vkazr0")
        parts = name.split("_")
        suffixes.append(parts[-1] if parts else name)
    return "_".join(suffixes)


def main():
    parser = argparse.ArgumentParser(description="Collect rollout results from folders.")
    parser.add_argument(
        "--folders",
        nargs="+",
        required=True,
        help="One or more rollout result folders to scan.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save output files.",
    )
    parser.add_argument(
        "--min-rate",
        type=float,
        default=0.25,
        help="Minimum pass rate for filtering (default: 0.25).",
    )
    parser.add_argument(
        "--max-rate",
        type=float,
        default=0.75,
        help="Maximum pass rate for filtering (default: 0.75).",
    )
    parser.add_argument(
        "--bin-size",
        type=float,
        default=0.1,
        help="Bin size for pass rate distribution (default: 0.1).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=64,
        help="Number of threads for parallel reward-file reads (default: 32). "
             "Increase for network filesystems with high latency.",
    )
    parser.add_argument(
        "--save-resolved-trajectories",
        action="store_true",
        default=False,
        help="If set, export every resolved trajectory into a single JSONL training file.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = make_output_name(args.folders)

    jsonl_path = output_dir / f"rollout_results_{suffix}.jsonl"
    grouped: dict | None = None

    if jsonl_path.exists():
        print(f"Found existing results file: {jsonl_path}")
        results = load_jsonl(str(jsonl_path))
        if args.save_resolved_trajectories:
            print(f"Scanning {len(args.folders)} folder(s) for resolved trajectory export...")
            grouped = collect_from_folders(args.folders)
    else:
        # Step 1 & 2: Collect and count
        print(f"Scanning {len(args.folders)} folder(s)...")
        grouped = collect_from_folders(args.folders)
        results = build_results(grouped, max_workers=args.workers)
        print(f"Found {len(results)} unique instance IDs.")

        # Save JSONL
        save_jsonl(results, str(jsonl_path))

    # Export every resolved trajectory into a single JSONL training file.
    if args.save_resolved_trajectories:
        assert grouped is not None
        rate_map = build_rate_map(results)
        resolved_jsonl_path = output_dir / f"resolved_trajectories_{suffix}.jsonl"
        export_resolved_jsonl(grouped, rate_map, str(resolved_jsonl_path),
                              max_workers=args.workers)

    # Step 3: Distribution and filtering
    print_rate_distribution(results, "reward_files", "trajectory_files",
                            " Submitted Rate Distribution ", bin_size=args.bin_size)
    print_rate_distribution(results, "resolved", "trajectory_files",
                            " Pass Rate Distribution ", bin_size=args.bin_size,
                            pass_at_ks=(3,))

    filtered = filter_instances_by_rate(results, args.min_rate, args.max_rate)
    print(f"\nInstances with pass rate in [{args.min_rate}, {args.max_rate}]: {len(filtered)}")

    filtered_path = output_dir / f"filtered_instances_{suffix}_rate_{args.min_rate}_{args.max_rate}.txt"
    with open(filtered_path, "w") as f:
        for iid in filtered:
            f.write(iid + "\n")
    print(f"Saved filtered instance IDs to {filtered_path}")

    # Randomly sample zero-accuracy instances (same count as filtered) and combine
    random.seed(42)
    zero_instances = get_zero_accuracy_instances(results)
    sample_count = min(len(filtered), len(zero_instances))
    sampled_zero = random.sample(zero_instances, sample_count)
    print(f"\nZero-accuracy instances available: {len(zero_instances)}, sampled: {sample_count}")

    combined = sorted(set(filtered + sampled_zero))
    combined_path = output_dir / f"filtered_instances_{suffix}_rate_{args.min_rate}_{args.max_rate}_with_zero.txt"
    with open(combined_path, "w") as f:
        for iid in combined:
            f.write(iid + "\n")
    print(f"Combined instances (filtered + sampled zero-accuracy): {len(combined)}")
    print(f"Saved combined instance IDs to {combined_path}")


if __name__ == "__main__":
    main()
