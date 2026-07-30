#!/usr/bin/env python3
"""Convert all distributed checkpoints (iter_XXXXXXX) under a directory to HuggingFace format."""

import argparse
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed


def find_iter_dirs(model_ckpt_dir: str):
    """Find all iter_XXXXXXX directories (excluding _hf ones) and return them sorted."""
    pattern = re.compile(r"^iter_\d+$")
    iter_dirs = []
    for name in os.listdir(model_ckpt_dir):
        full_path = os.path.join(model_ckpt_dir, name)
        if os.path.isdir(full_path) and pattern.match(name):
            iter_dirs.append(name)
    return sorted(iter_dirs)


def convert_checkpoint(input_dir: str, output_dir: str, origin_hf_dir: str):
    """Run the conversion command for a single checkpoint."""
    cmd = [
        "python", "tools/convert_torch_dist_to_hf.py",
        "--input-dir", input_dir,
        "--output-dir", output_dir,
        "--origin-hf-dir", origin_hf_dir,
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=os.path.expanduser("~/slime"), env={**os.environ, "PYTHONPATH": "/root/Megatron-LM"})
    return result.returncode


def _convert_one_task(task):
    """Worker function: convert a single checkpoint and optionally remove the input dir."""
    input_dir, output_dir, origin_hf_dir, remove_after = task
    iter_name = os.path.basename(input_dir)
    print(f"\n{'='*60}\nConverting {iter_name} ...\n  Input:  {input_dir}\n  Output: {output_dir}\n{'='*60}")
    ret = convert_checkpoint(input_dir, output_dir, origin_hf_dir)
    if ret == 0 and remove_after:
        print(f"Removing original checkpoint: {input_dir}")
        shutil.rmtree(input_dir)
    return input_dir, ret


def main():
    parser = argparse.ArgumentParser(description="Convert all dist checkpoints to HuggingFace format.")
    parser.add_argument("--model-ckpt-dir", type=str, required=False, nargs="+", default=[],
                        help="One or more directories containing iter_XXXXXXX subdirectories.")
    parser.add_argument("--model-ckpt-parent-dir", type=str, required=False, nargs="+", default=[],
                        help="One or more parent directories under which to search for subdirectories "
                             "matching --ending-flag.")
    parser.add_argument("--ending-flag", type=str, required=False, nargs="+", default=[],
                        help="One or more suffixes; subdirectories of --model-ckpt-parent-dir ending with any of these "
                             "are treated as additional --model-ckpt-dir entries.")
    parser.add_argument("--origin-hf-dir", type=str, required=True,
                        help="Path to the original HuggingFace model directory.")
    parser.add_argument("--remove-after-conversion", action="store_true", default=False,
                        help="Remove the original iter_XXXXXXX directory after successful conversion.")
    parser.add_argument("--max-parallel", type=int, default=8,
                        help="Number of parallel conversion processes (default: 8).")
    args = parser.parse_args()

    model_ckpt_dirs = [os.path.abspath(p) for p in args.model_ckpt_dir]

    if args.ending_flag:
        if not args.model_ckpt_parent_dir:
            print("Error: --ending-flag requires --model-ckpt-parent-dir.")
            sys.exit(1)
        for raw_parent in args.model_ckpt_parent_dir:
            parent_dir = os.path.abspath(raw_parent)
            if not os.path.isdir(parent_dir):
                print(f"Error: --model-ckpt-parent-dir does not exist: {parent_dir}")
                sys.exit(1)
            matched = []
            for name in sorted(os.listdir(parent_dir)):
                full_path = os.path.join(parent_dir, name)
                if os.path.isdir(full_path) and any(name.endswith(flag) for flag in args.ending_flag):
                    matched.append(full_path)
            print(f"Found {len(matched)} subdirectory(ies) under {parent_dir} matching ending flags {args.ending_flag}:")
            for p in matched:
                print(f"  {p}")
            model_ckpt_dirs.extend(matched)
    elif args.model_ckpt_parent_dir:
        print("Error: --model-ckpt-parent-dir requires --ending-flag.")
        sys.exit(1)

    # De-duplicate while preserving order.
    seen = set()
    deduped = []
    for d in model_ckpt_dirs:
        if d not in seen:
            seen.add(d)
            deduped.append(d)
    model_ckpt_dirs = deduped

    if not model_ckpt_dirs:
        print("Error: no model checkpoint directories specified. Use --model-ckpt-dir or --model-ckpt-parent-dir/--ending-flag.")
        sys.exit(1)

    for d in model_ckpt_dirs:
        if not os.path.isdir(d):
            print(f"Error: model checkpoint directory does not exist: {d}")
            sys.exit(1)

    # Collect all conversion tasks across all model_ckpt_dirs.
    tasks = []
    for model_ckpt_dir in model_ckpt_dirs:
        iter_dirs = find_iter_dirs(model_ckpt_dir)
        print(f"\n##### Scanning {model_ckpt_dir} #####")
        if not iter_dirs:
            print(f"No iter_XXXXXXX directories found in {model_ckpt_dir}")
            continue
        print(f"Found {len(iter_dirs)} checkpoint(s): {iter_dirs}")

        for iter_name in iter_dirs:
            input_dir = os.path.join(model_ckpt_dir, iter_name)
            output_dir = os.path.join(model_ckpt_dir, f"{iter_name}_hf")

            if os.path.isdir(output_dir):
                print(f"Skipping {iter_name}: output directory already exists at {output_dir}")
                continue

            tasks.append((input_dir, output_dir, args.origin_hf_dir, args.remove_after_conversion))

    total_count = len(tasks)
    if total_count == 0:
        print("\nNo checkpoints to convert.")
        return

    max_workers = max(1, min(args.max_parallel, total_count))
    print(f"\nLaunching {total_count} conversion task(s) with {max_workers} parallel worker(s).")

    total_failed = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_convert_one_task, t): t for t in tasks}
        for fut in as_completed(futures):
            try:
                input_dir, ret = fut.result()
            except Exception as e:
                t = futures[fut]
                print(f"Error: worker raised exception for {t[0]}: {e}")
                total_failed.append(t[0])
                continue
            if ret != 0:
                print(f"Error: conversion failed for {input_dir} (exit code {ret})")
                total_failed.append(input_dir)

    print(f"\n{'='*60}")
    print(f"Done. {total_count - len(total_failed)}/{total_count} converted successfully.")
    if total_failed:
        print(f"Failed: {total_failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
