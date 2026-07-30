#!/usr/bin/env python3
"""
Filter out already-used data instances from a full dataset.

Reads two JSONL files:
  1. Full dataset
  2. Already-used dataset
Removes instances from the full set whose 'instance_id' appears in the
already-used set, and writes the remaining instances to an output JSONL file.
"""

import argparse
import json


def load_instance_ids(path: str) -> set:
    """Load all instance_id values from a JSONL file."""
    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            instance_id = record.get("instance_id") or record.get("metadata", {}).get("instance_id")
            if instance_id is not None:
                ids.add(instance_id)
    return ids


def filter_data(full_path: str, used_path: str, output_path: str) -> None:
    used_ids = load_instance_ids(used_path)
    print(f"Loaded {len(used_ids)} already-used instance IDs from {used_path}")

    total = 0
    kept = 0
    with open(full_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            record = json.loads(line)
            instance_id = record.get("instance_id") or record.get("metadata", {}).get("instance_id")
            if instance_id not in used_ids:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1

    removed = total - kept
    print(f"Total: {total}, Removed: {removed}, Remaining: {kept}")
    print(f"Output written to {output_path}")


def keeps_data(full_path: str, keep_path: str, output_path: str) -> None:
    if keep_path.endswith(".jsonl"):
        keep_ids = load_instance_ids(keep_path)
    else:
        keep_ids = set()
        with open(keep_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    keep_ids.add(line)
    print(f"Loaded {len(keep_ids)} instance IDs to keep from {keep_path}")

    total = 0
    kept = 0
    with open(full_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            record = json.loads(line)
            instance_id = record.get("instance_id") or record.get("metadata", {}).get("instance_id")
            if instance_id in keep_ids:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1

    print(f"Total: {total}, Kept: {kept}, Skipped: {total - kept}")
    print(f"Output written to {output_path}")


# # Keep only specified instances
# python filter_used_data.py keep --full data.jsonl --keep ids.txt --output out.jsonl

# # Filter out used instances (existing behavior)
# python filter_used_data.py filter --full data.jsonl --used used.jsonl --output out.jsonl

def main():
    parser = argparse.ArgumentParser(
        description="Filter JSONL dataset by instance IDs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    filter_parser = subparsers.add_parser("filter", help="Remove already-used instances.")
    filter_parser.add_argument("--full", required=True, help="Path to the full dataset JSONL file.")
    filter_parser.add_argument("--used", required=True, help="Path to the already-used dataset JSONL file.")
    filter_parser.add_argument("--output", required=True, help="Path to write the filtered output JSONL file.")

    keep_parser = subparsers.add_parser("keep", help="Keep only specified instances.")
    keep_parser.add_argument("--full", required=True, help="Path to the full dataset JSONL file.")
    keep_parser.add_argument("--keep", required=True, help="Path to a text file with one instance_id per line, or a JSONL file with 'instance_id' fields.")
    keep_parser.add_argument("--output", required=True, help="Path to write the filtered output JSONL file.")

    args = parser.parse_args()

    if args.command == "filter":
        filter_data(args.full, args.used, args.output)
    elif args.command == "keep":
        keeps_data(args.full, args.keep, args.output)


if __name__ == "__main__":
    main()
