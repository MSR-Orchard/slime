import json
import glob
import argparse
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

def get_time_duration_from_json(json_file):
    llm_inference_time_list = []
    env_execution_time_list = []
    with open(json_file, "r") as f:
        data = json.load(f)
        env_creation_time = data.get("env_creation_time", 0.0)
        exit_status = data.get("exit_status", None)
        if "messages" not in data:
            print(f"Warning: 'messages' key not found in {json_file}, skipping.")
            input("continue?")
            return llm_inference_time_list, env_execution_time_list, 0.0, env_creation_time, exit_status
        for i, message in enumerate(data["messages"]):
            if i == 0:
                continue
            starttime = data["messages"][i-1]["timestamp"]
            endtime = message["timestamp"]
            duration = endtime - starttime
            if message["role"] == "assistant":
                llm_inference_time_list.append(duration)
            elif message["role"] in ["user", "tool"]:
                env_execution_time_list.append(duration)
    
    entire_duration = data["messages"][-1]["timestamp"] - data["messages"][0]["timestamp"]

    return llm_inference_time_list, env_execution_time_list, entire_duration, env_creation_time, exit_status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze time durations from SWE rollout JSON files.")
    parser.add_argument("--json_folder", type=str, help="Path to the folder containing SWE rollout JSON files.")
    args = parser.parse_args()

    all_json_files = glob.glob(str(Path(args.json_folder) / "*.json"))
    all_json_files = [f for f in all_json_files if "_rewards" not in f]
    if len(all_json_files) > 10000:
        random.seed(42)
        json_files = random.sample(all_json_files, 10000)
        print(f"Randomly sampled 10000 files out of {len(all_json_files)} total files.")
    else:
        json_files = all_json_files
    all_llm_inference_times = []
    all_env_execution_times = []
    all_entire_durations = []
    all_env_creation_times = []
    all_exit_statuses = []

    json_file2duration = {}
    entire_duration2json_file = {}

    with ThreadPoolExecutor(max_workers=64) as executor:
        results = list(executor.map(get_time_duration_from_json, json_files))

    for json_file, result in zip(json_files, results):
        llm_inference_time_list, env_execution_time_list, entire_duration, env_creation_time, exit_status = result
        all_llm_inference_times.extend(llm_inference_time_list)
        all_env_execution_times.extend(env_execution_time_list)
        all_entire_durations.append(entire_duration)
        all_env_creation_times.append(env_creation_time)
        all_exit_statuses.append(exit_status)
        json_file2duration[json_file] = {
            "llm_inference_time_list": llm_inference_time_list,
            "env_execution_time_list": env_execution_time_list,
            "entire_duration": entire_duration,
            "env_creation_time": env_creation_time,
        }
        entire_duration2json_file[entire_duration] = json_file
        

    average_llm_inference_time = sum(all_llm_inference_times) / len(all_llm_inference_times) if all_llm_inference_times else 0
    average_env_execution_time = sum(all_env_execution_times) / len(all_env_execution_times) if all_env_execution_times else 0
    average_entire_duration = sum(all_entire_durations) / len(all_entire_durations) if all_entire_durations else 0
    average_env_creation_time = sum(all_env_creation_times) / len(all_env_creation_times) if all_env_creation_times else 0

    print(f"Average LLM inference time: {average_llm_inference_time:.2f} s")
    print(f"Average environment execution time: {average_env_execution_time:.2f} s")
    print(f"Average entire duration: {average_entire_duration:.2f} s")
    print(f"Average env creation time: {average_env_creation_time:.2f} s")

    # get max time for each category and print the corresponding json file
    max_llm_inference_time = max(all_llm_inference_times)
    max_env_execution_time = max(all_env_execution_times)
    max_entire_duration = max(all_entire_durations)
    max_env_creation_time = max(all_env_creation_times)

    for json_file, duration_dict in json_file2duration.items():
        if max_llm_inference_time in duration_dict["llm_inference_time_list"]:
            print(f"Max LLM inference time: {max_llm_inference_time:.2f} s, JSON file: {json_file}")
        if max_env_execution_time in duration_dict["env_execution_time_list"]:
            print(f"Max environment execution time: {max_env_execution_time:.2f} s, JSON file: {json_file}")
        if max_entire_duration == duration_dict["entire_duration"]:
            print(f"Max entire duration: {max_entire_duration:.2f} s, JSON file: {json_file}")
        if max_env_creation_time == duration_dict["env_creation_time"]:
            print(f"Max env creation time: {max_env_creation_time:.2f} s, JSON file: {json_file}")
    
    # print the distribution of LLM inference time and environment execution time (print into bins of 60 seconds)
    llm_bins = [0] * 20  # 0-95+ seconds in 5-second bins
    env_bins = [0] * 12
    entire_bins = [0] * 11
    env_bin_index2time_list = {}
    entire_bin_index2time_list = {}

    for time in all_llm_inference_times:
        llm_bins[min(int(time / 5), 19)] += 1

    for time in all_env_execution_times:
        bin_index = min(int(time / 30), 11)
        env_bins[bin_index] += 1
        if bin_index not in env_bin_index2time_list:
            env_bin_index2time_list[bin_index] = []
        env_bin_index2time_list[bin_index].append(time)
    
    for time in all_entire_durations:
        bin_index = min(int(time / 300), 10)
        entire_bins[bin_index] += 1
        if bin_index not in entire_bin_index2time_list:
            entire_bin_index2time_list[bin_index] = []
        entire_bin_index2time_list[bin_index].append(time)

    print("LLM inference time distribution (5-second bins):")
    acc_percentage = 0
    for i, count in enumerate(llm_bins):
        percentage = (count / len(all_llm_inference_times) * 100)
        acc_percentage += percentage
        if i == 19:
            print(f"  {i*5}+ seconds: {count} samples ({percentage:.2f}%, acc: {acc_percentage:.2f}%)")
        else:
            print(f"  {i*5}-{(i+1)*5} seconds: {count} samples ({percentage:.2f}%, acc: {acc_percentage:.2f}%)")
    
    input("continue?")
    
    print("Environment execution time distribution (30-second bins):")
    acc_percentage = 0
    for i, count in enumerate(env_bins):
        percentage = (count / len(all_env_execution_times) * 100)
        acc_percentage += percentage
        if i == 11:
            print(f"  {i*30}+ seconds: {count} samples ({percentage:.2f}%, acc: {acc_percentage:.2f}%)")
            if i in env_bin_index2time_list:
                print(f"    Times: {env_bin_index2time_list[i]}")
        else:
            print(f"  {i*30}-{(i+1)*30} seconds: {count} samples ({percentage:.2f}%, acc: {acc_percentage:.2f}%)")
    
    input("continue?")

    print("Entire duration distribution (300-second bins):")
    acc_percentage = 0
    for i, count in enumerate(entire_bins):
        percentage = (count / len(all_entire_durations) * 100)
        acc_percentage += percentage
        if i == 10:
            print(f"  {i*300}+ seconds: {count} samples ({percentage:.2f}%, acc: {acc_percentage:.2f}%)")
            if i in entire_bin_index2time_list:
                print("    Corresponding JSON files:")
                for time in entire_bin_index2time_list[i]:
                    json_file = entire_duration2json_file[time]
                    print(f"      {json_file}: {time:.2f} seconds")
        else:
            print(f"  {i*300}-{(i+1)*300} seconds: {count} samples ({percentage:.2f}%, acc: {acc_percentage:.2f}%)")

    input("continue?")

    env_creation_bins = [0] * 11
    env_creation_bin_index2time_list = {}
    for time in all_env_creation_times:
        bin_index = min(int(time / 60), 10)
        env_creation_bins[bin_index] += 1
        if bin_index not in env_creation_bin_index2time_list:
            env_creation_bin_index2time_list[bin_index] = []
        env_creation_bin_index2time_list[bin_index].append(time)

    print("Env creation time distribution (60-second bins):")
    acc_percentage = 0
    for i, count in enumerate(env_creation_bins):
        percentage = (count / len(all_env_creation_times) * 100) if all_env_creation_times else 0
        acc_percentage += percentage
        if i == 10:
            print(f"  {i*60}+ seconds: {count} samples ({percentage:.2f}%, acc: {acc_percentage:.2f}%)")
            if i in env_creation_bin_index2time_list:
                print("    Corresponding JSON files:")
                for time in env_creation_bin_index2time_list[i]:
                    for jf, dd in json_file2duration.items():
                        if dd["env_creation_time"] == time:
                            print(f"      {jf}: {time:.2f} seconds")
        else:
            print(f"  {i*60}-{(i+1)*60} seconds: {count} samples ({percentage:.2f}%, acc: {acc_percentage:.2f}%)")

    input("continue?")

    exit_status_counts = {}
    for status in all_exit_statuses:
        key = status if status is not None else "None"
        exit_status_counts[key] = exit_status_counts.get(key, 0) + 1

    total_exit_statuses = len(all_exit_statuses)
    print(f"Exit status distribution (total {total_exit_statuses} samples):")
    for status, count in sorted(exit_status_counts.items(), key=lambda x: -x[1]):
        percentage = (count / total_exit_statuses * 100) if total_exit_statuses else 0
        print(f"  {status}: {count} samples ({percentage:.2f}%)")

    input("continue?")

    # Scan all jsonl files in the target folder. Each jsonl file contains a list
    # of json objects with a "reward" field. Compute the per-file percentage of
    # entries with reward == 1.0, then report the distribution of these
    # percentages across all jsonl files in 0.125-wide buckets.
    jsonl_files = glob.glob(str(Path(args.json_folder) / "*.jsonl"))
    print(f"\nFound {len(jsonl_files)} jsonl files in {args.json_folder}")

    per_file_reward1_percentages = []
    def _scan_jsonl(jsonl_file):
        total = 0
        reward1_count = 0
        with open(jsonl_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                total += 1
                if obj.get("reward") == 1.0:
                    reward1_count += 1
        if total == 0:
            return None
        return reward1_count / total

    with ThreadPoolExecutor(max_workers=32) as executor:
        for p in executor.map(_scan_jsonl, jsonl_files):
            if p is not None:
                per_file_reward1_percentages.append(p)

    num_files = len(per_file_reward1_percentages)
    print(f"reward=1.0 percentage distribution across {num_files} jsonl files (0.125-wide buckets):")
    bucket_edges = [i * 0.125 for i in range(9)]  # 0.0, 0.125, ..., 1.0
    bucket_counts = [0] * 8
    for p in per_file_reward1_percentages:
        # Map p in [0, 1] to bucket index [0, 7]; include 1.0 in the last bucket
        idx = min(int(p / 0.125), 7)
        bucket_counts[idx] += 1

    acc_percentage = 0
    for i, count in enumerate(bucket_counts):
        lo = bucket_edges[i]
        hi = bucket_edges[i + 1]
        percentage = (count / num_files * 100) if num_files else 0
        acc_percentage += percentage
        right_bracket = "]" if i == 7 else ")"
        print(f"  [{lo:.3f}, {hi:.3f}{right_bracket}: {count} files ({percentage:.2f}%, acc: {acc_percentage:.2f}%)")

    exact_zero = sum(1 for p in per_file_reward1_percentages if p == 0.0)
    exact_one = sum(1 for p in per_file_reward1_percentages if p == 1.0)
    zero_pct = (exact_zero / num_files * 100) if num_files else 0
    one_pct = (exact_one / num_files * 100) if num_files else 0
    print(f"  exactly 0.0: {exact_zero} files ({zero_pct:.2f}%)")
    print(f"  exactly 1.0: {exact_one} files ({one_pct:.2f}%)")
