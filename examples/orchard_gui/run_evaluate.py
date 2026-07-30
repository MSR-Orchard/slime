"""
Evaluate browser agent success rate on WebVoyager validation set.

Runs generate_turn_sample for each task, then uses the LLM judge
to determine success. Supports parallel evaluation with a bounded number
of concurrent environments.

Usage:
    python run_evaluate.py --task-file examples/orchard_gui/env/tasks/webvoyager_val.jsonl \
                           --n-parallel 4 \
                           --sglang-ip localhost --sglang-port 30000
"""

import argparse
import asyncio
import glob
import json
import os
import re
import sys
import datetime
import yaml
from urllib.parse import urlparse

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dataclasses import dataclass

from examples.orchard_gui.generate_browser import generate_turn_sample
from slime.utils.types import Sample
from slime.utils.http_utils import init_http_client


@dataclass
class RolloutArgs:
    """Standalone eval arguments (extracted from the removed run_rollouts.py)."""
    sglang_router_ip: str = "localhost"
    sglang_router_port: int = 30000
    partial_rollout: bool = False
    hf_checkpoint: str = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    sglang_server_concurrency: int = 16
    rollout_num_gpus: int = 8 # unused, rollouts are executed via HTTP calls to sglang server which handles GPU allocation internally based on its own config and current load
    rollout_num_gpus_per_engine: int = 8 # unused, see above
    rollout_temperature: float = 0.6
    rollout_top_p: float = 0.95
    rollout_top_k: int = 20
    rollout_min_p: float = 0.0
    rollout_presence_penalty: float = 0.0
    rollout_repetition_penalty: float = 1.0
    rollout_max_response_len: int = 4096
    rollout_stop: list = None
    rollout_stop_token_ids: list = None
    rollout_skip_special_tokens: bool = False
    use_distributed_post: bool = False
    sglang_dp_size: int = 1

    # --- browser_training_config.yaml params ---
    rollout_max_context_len: int = 32768
    max_consecutive_parse_failures: int = 3
    max_steps: int = 30
    context_num_screenshots: int = 1
    include_tool_response: bool = True
    include_history_thoughts: bool = True
    path_to_save_generated_samples: str = ""
    # Per-await timeouts (seconds) in generate_browser.py
    env_init_timeout: int = 300
    inference_timeout: int = 300
    env_step_timeout: int = 120
    env_exit_timeout: int = 60
    # Overall wall-clock cap for one rollout (safety net on top of the per-await
    # timeouts above). None = disabled.
    rollout_task_timeout_secs: "float | None" = None
    # Reward judge config
    judge_max_attached_imgs: int = 3
    judge_api_mode: str = "token"
    judge_api_model: str = "gpt-4.1"
    judge_prompt_variant: str = "action_history"
    judge_timeout_secs: float = 120
    log_judge_output: bool = False
    debug_trace_prob: float = 0.0

    # If non-empty, overrides `mode` loaded from env/config.yaml
    # (one of "local", "remote", "sandbox", "browser-use").
    env_mode: str = ""


def load_tasks_from_jsonl(task_file: str, task_start: int = 0, task_end: int = -1, shuffle: bool = False) -> list[dict]:
    """Load tasks from a JSONL file and fix up metadata fields.

    Supports both schemas: lines with a prebuilt ``metadata`` dict (pod-viable
    style) and raw benchmark lines (examples/orchard_gui/data/*.jsonl), where
    ``metadata`` is synthesized from the flat fields (intent <- task_name,
    start_url <- website).
    """
    tasks = []
    with open(task_file, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            task_data = json.loads(line)
            if "metadata" not in task_data:
                md = dict(task_data)
                md.setdefault("intent", task_data.get("task_name", ""))
                md.setdefault("start_url", task_data.get("website", ""))
                task_data["metadata"] = md
            tasks.append(task_data)

    if shuffle:
        import random
        random.shuffle(tasks)

    # Slice by range
    if task_end > 0:
        tasks = tasks[task_start:task_end]
    else:
        tasks = tasks[task_start:]

    return tasks


def load_yaml_config(config_path: str) -> dict:
    """Load a YAML config file and return as a dict."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}
    return config


# ==============================================================================
# Captcha detection (shared with correct_captcha_via_full_traj.py)
# ==============================================================================
#
# Two-case rule:
#   1. last assistant turn mentions a captcha keyword → the run ended on
#      a captcha.
#   2. a middle assistant turn mentions a captcha keyword AND a later
#      (or same) turn issues a ``goto_url`` whose host differs from
#      ``start_url`` → the agent saw the captcha and pivoted away.
# Same-host goto_urls (e.g. moving within Amazon) don't count, since
# they typically aren't a captcha-driven pivot.
CAPTCHA_KEYWORDS = [
    "captcha", "anti-bot", "cloudflare", "blocked", "forbidden",
    "access denied", "bot verification", "verify you are human",
    "rate limit", "restricted",
]

# A saved result file's `prompt` is the full chat-template-formatted
# history including system/user/tool turns. Page content (a11ytree,
# observations) routinely contains words like "blocked"/"restricted"/
# "forbidden", so scanning the whole prompt produces massive false
# positives. Extract only assistant turns (Qwen chat format).
_ASSISTANT_TURN_RE = re.compile(
    r"<\|im_start\|>assistant\s*\n(.*?)<\|im_end\|>", re.DOTALL,
)
# Tool calls are emitted as `<tool_call>{...}</tool_call>`; capture the
# JSON payload so we can read both the action name and its url argument.
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _assistant_turns_from_result(r: dict) -> list[str]:
    """Return assistant turns in trajectory order from a saved result
    dict. ``response`` is the final turn; the prompt holds every prior
    assistant turn."""
    prompt = r.get("prompt") or ""
    response = r.get("response") or ""
    turns = _ASSISTANT_TURN_RE.findall(prompt)
    if response:
        turns.append(response)
    return turns


def _mentions_captcha(text: str) -> bool:
    text = text.lower()
    return any(kw in text for kw in CAPTCHA_KEYWORDS)


def _extract_goto_urls(turn_text: str) -> list[str]:
    """Return the URL argument of every ``goto_url`` tool call in *turn_text*."""
    urls: list[str] = []
    for m in _TOOL_CALL_RE.finditer(turn_text):
        try:
            call = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if not isinstance(call, dict) or call.get("name") != "goto_url":
            continue
        args = call.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                continue
        if isinstance(args, dict):
            url = args.get("url")
            if isinstance(url, str):
                urls.append(url)
    return urls


def _norm_host(url: str) -> str:
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""
    if host.startswith("www."):
        host = host[4:]
    return host


def _same_site(a: str, b: str) -> bool:
    """Treat hosts as same-site if equal or one is a subdomain of the
    other (after stripping the leading ``www.``). Empty hosts (relative
    URLs, missing start_url) → conservatively same-site."""
    if not a or not b:
        return True
    if a == b:
        return True
    return a.endswith("." + b) or b.endswith("." + a)


def _hit_captcha_from_turns(turns: list[str], start_url: str) -> bool:
    """Apply the two-case rule to an ordered list of assistant-turn texts."""
    if not turns:
        return False

    # Case 1: last turn mentions captcha → run ended on a captcha.
    if _mentions_captcha(turns[-1]):
        return True

    # Case 2: a middle turn mentions captcha AND a later (or same) turn
    # navigates to a different host than start_url.
    start_host = _norm_host(start_url or "")
    for i, turn in enumerate(turns[:-1]):
        if not _mentions_captcha(turn):
            continue
        for j in range(i, len(turns)):
            for url in _extract_goto_urls(turns[j]):
                if not _same_site(_norm_host(url), start_host):
                    return True
    return False


def hit_captcha_from_messages(messages: list[dict], start_url: str) -> bool:
    """Two-case rule applied to an in-memory chat messages list (the
    runtime form held in ``sample.metadata['messages']``)."""
    turns: list[str] = []
    for m in messages or []:
        if m.get("role") != "assistant":
            continue
        c = m.get("content", "")
        if isinstance(c, list):
            c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
        turns.append(c or "")
    return _hit_captcha_from_turns(turns, start_url)

# ==============================================================================
# End of Captcha detection
# ==============================================================================

def _load_reward_func(protocol: str):
    """Return the reward_func selected by --eval-protocol."""
    if protocol == "default":
        from examples.orchard_gui.reward_browser import reward_func
    elif protocol == "webvoyager":
        from examples.orchard_gui.eval.reward_webvoyager import reward_func
    elif protocol == "online_mind2web":
        from examples.orchard_gui.eval.reward_online_mind2web import reward_func
    elif protocol == "deepshop":
        from examples.orchard_gui.eval.reward_deepshop import reward_func
    else:
        raise ValueError(f"Unknown --eval-protocol: {protocol}")
    return reward_func



async def evaluate_all(args: RolloutArgs, tasks: list[dict], n_parallel: int,
                       sampling_params: dict, reward_func, output_path: str = "") -> list[dict]:
    """Evaluate all tasks with bounded parallelism."""
    sem = asyncio.Semaphore(n_parallel)

    async def run_with_sem(task_meta):
        task_id = task_meta.get("task_id", "unknown")
        try:
            # Phase 1: Generate (holds main semaphore for sandbox access)
            async with sem:
                print(f"[START] {task_id}")
                input_sample = Sample(
                    prompt=task_meta.get("intent", ""),
                    metadata=dict(task_meta),
                )
                turn_result = await generate_turn_sample(args, input_sample, sampling_params)

            # Phase 2: Reward (outside main semaphore to avoid nested-semaphore starvation)
            # Timeout guards against judge API retries blocking the last few tasks.
            try:
                await asyncio.wait_for(reward_func(args, turn_result), timeout=300)
            except asyncio.TimeoutError:
                print(f"[WARN]  {task_id}: reward_func timed out after 300s, scoring as None (judge-side failure)")
                for s in turn_result:
                    s.reward = None
                    s.remove_sample = True
                    s.metadata["reward"] = {"judge": None, "combined": None, "judge_text": "reward timeout", "judge_timeout": True}
            last = turn_result[-1]

            # Two-case captcha rule (see correct_captcha_via_full_traj.py):
            #   1. last assistant turn mentions a captcha keyword, OR
            #   2. a middle turn mentions a captcha keyword AND a later
            #      goto_url navigates to a different host than start_url.
            hit_captcha = hit_captcha_from_messages(
                last.metadata.get("messages", []),
                task_meta.get("start_url", ""),
            )

            result = {
                "task_id": task_id,
                "status": last.status.value if hasattr(last.status, 'value') else str(last.status),
                "reward": last.reward,
                "total_steps": last.metadata.get("total_steps", -1),
                "terminate_reason": last.metadata.get("terminate_reason", ""),
                "hit_captcha": hit_captcha,
                "reward_meta": last.metadata.get("reward", {}) or {},
                "prompt": last.prompt,
                "response": last.response,
                "intent": task_meta.get("intent", ""),
                "start_url": task_meta.get("start_url", ""),
            }
            print(f"[DONE]  {task_id} -> [reward={result['reward']}, status={result['status']}, steps={result['total_steps']}]")
            if output_path:
                safe_id = task_id.replace("/", "_")
                with open(os.path.join(output_path, f"results_task_{safe_id}.json"), "w") as f:
                    json.dump(result, f, indent=2, default=str)
            return result
        except Exception as e:
            print(f"[ERROR] {task_id}: {e}")
            import traceback
            traceback.print_exc()
            return {
                "task_id": task_id,
                "status": "error",
                "reward": -1.0,
                "total_steps": 0,
                "terminate_reason": f"exception: {e}",
                "response": "",
                "intent": task_meta.get("intent", ""),
                "start_url": task_meta.get("start_url", ""),
            }

    from tqdm import tqdm
    total = len(tasks)
    pbar = tqdm(total=total, desc="Evaluating", unit="task",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}\n")
    succeeded = 0
    failed = 0
    judge_err = 0

    async def _tracked(coro):
        nonlocal succeeded, failed, judge_err
        result = await coro
        reward = result.get("reward", -1)
        if reward is None:
            judge_err += 1
        elif reward == 1.0:
            succeeded += 1
        else:
            failed += 1
        pbar.set_postfix_str(f"succ={succeeded} fail={failed} judge_err={judge_err}\n", refresh=False)
        pbar.update(1)
        return result

    coros = [_tracked(run_with_sem(t["metadata"])) for t in tasks]
    results = await asyncio.gather(*coros)
    pbar.close()
    return list(results)


def get_eval_summary(results: list[dict]) -> str:
    """Build evaluation summary."""
    total = len(results)
    if total == 0:
        return "No results to summarize."

    stats = {"succeeded": []}
    judge_errors: list[dict] = []
    for r in results:
        if r["reward"] is None:
            judge_errors.append(r)
        elif r["reward"] == 1.0:
            stats["succeeded"].append(r)
        else:
            stats.setdefault(r["status"], []).append(r)

    status_names = "\t".join([f"{k[:4]}." for k in stats.keys()])
    status_counts = "\t".join(str(len(v)) for v in stats.values())
    summary = f"Successes ({len(stats['succeeded'])}): {[r['task_id'] for r in stats['succeeded']]}\n"

    for status, stat_results in stats.items():
        if status == "succeeded":
            continue
        captcha_count = sum(1 for r in stat_results if r.get("hit_captcha", False))
        summary += f"\nDetails for {status} tasks ({captcha_count}/{len(stat_results)} hit captcha):\n"
        for r in stat_results:
            terminate_reason = str(r['terminate_reason'])[:20]
            summary += f"- Task {r['task_id']}: reward={r['reward']}, steps={r['total_steps']}, terminate_reason={terminate_reason}, hit_captcha={r.get('hit_captcha', False)}\n"

    captcha_results = [r for r in results if r.get("hit_captcha", False) and r.get("reward") != 1.0]
    if captcha_results:
        summary += f"\nDetails for captcha tasks ({len(captcha_results)}):\n"
        for r in captcha_results:
            terminate_reason = str(r['terminate_reason'])[:20]
            summary += f"- Task {r['task_id']}: reward={r['reward']}, steps={r['total_steps']}, terminate_reason={terminate_reason}, status={r.get('status')}\n"

    if judge_errors:
        summary += f"\nDetails for judge-error tasks ({len(judge_errors)}):\n"
        for r in judge_errors:
            terminate_reason = str(r['terminate_reason'])[:20]
            summary += f"- Task {r['task_id']}: status={r.get('status')}, steps={r['total_steps']}, terminate_reason={terminate_reason}\n"

    # "Informative" = tasks where we could fairly evaluate the agent.
    # Exclude judge errors (judge call/parse failed), captcha hits, and aborts (env-side failures). Use a set on task_id to avoid double-
    # counting tasks that fall into more than one bucket (e.g. an aborted task that also hit a captcha).
    uninformative_ids = (
        {r["task_id"] for r in judge_errors}
        | {r["task_id"] for r in captcha_results}
        | {r["task_id"] for r in stats.get("aborted", [])}
    )
    informative_total = total - len(uninformative_ids)

    # Step-count stats exclude aborted tasks (env-side failures yield meaningless step counts).
    def _avg(xs):
        return sum(xs) / len(xs) if xs else 0.0
    def _steps(rs):
        return [r["total_steps"] for r in rs
                if r.get("status") != "aborted"
                and isinstance(r.get("total_steps"), int) and r["total_steps"] >= 0]
    all_steps = _steps(results)
    succ_steps = _steps(stats["succeeded"])
    info_steps = _steps([r for r in results if r["task_id"] not in uninformative_ids])

    summary += (f"\n------------------------------\n"
               f"Total tasks: {total} (informative: {informative_total}, judge_error: {len(judge_errors)})\n"
               f"Success rate w/o judge_error|captcha|aborted: {len(stats['succeeded']) / max(informative_total, 1) * 100:.1f}%\n"
               f"Success rate: {len(stats['succeeded']) / max(total, 1) * 100:.1f}%\n"
               f"Avg steps (excl. aborted): all={_avg(all_steps):.1f} (n={len(all_steps)}), "
               f"succeeded={_avg(succ_steps):.1f} (n={len(succ_steps)}), "
               f"non-captcha-juedge_error-aborted={_avg(info_steps):.1f} (n={len(info_steps)})\n"
               f"{status_names}\tcaptcha\tjudge_err\t|\tuninfo_uniq\n"
               f"{status_counts}\t{len(captcha_results)}\t{len(judge_errors)}\t|\t{len(uninformative_ids)}\n"
               f"uninformative present: {len(uninformative_ids) > 0}\n"
               f"------------------------------\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate browser agent success rate")
    parser.add_argument("--config", type=str, default="examples/orchard_gui/browser_training_config.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--output", type=str, default="", help="Output dir. Auto-generated when empty.")
    parser.add_argument("--task-file", type=str, default="examples/orchard_gui/env/tasks/webvoyager_val.jsonl",
                        help="Path to JSONL task file")
    parser.add_argument("--task-start", type=int, default=1)
    parser.add_argument("--task-end", type=int, default=2, help="End index (exclusive). -1 = all")
    parser.add_argument("--n-parallel", type=int, default=16,
                        help="Max number of concurrent browser environments")
    parser.add_argument("--hf-checkpoint", type=str, default="Qwen/Qwen3-VL-30B-A3B-Thinking")

    parser.add_argument("--eval-protocol", type=str, default="default",
                        choices=["default", "webvoyager", "online_mind2web", "deepshop"],
                        help="Reward protocol: 'default' = reward_browser (format + judge); "
                             "'webvoyager' = reward_webvoyager (FARA-style pure judge); "
                             "'online_mind2web' = reward_online_mind2web (FARA AgentTrek single-call); "
                             "'deepshop' = reward_deepshop (Molmo Web structured-output judge).")
    parser.add_argument("--env-mode", type=str, default="",
                        choices=["", "local", "remote", "sandbox", "browser-use"],
                        help="Override env mode from env/config.yaml. Empty = use yaml.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle tasks before evaluation")
    parser.add_argument("--save_sample", action="store_true", help="Save generated samples for each task")
    args = parser.parse_args()

    # Resolve output dir.
    if not args.output:
        expt_name = f"eval_{args.task_file.split('/')[-1].split('.')[0]}_{args.eval_protocol}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        args.output = os.path.join(args.hf_checkpoint, expt_name)
    os.makedirs(args.output, exist_ok=True)
    print(f"Output directory: {args.output}")

    # Tee stdout/stderr to both terminal and file
    log_path = os.path.join(args.output, "std_output.txt")
    log_file = open(log_path, "w")

    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()
        def flush(self):
            for s in self.streams:
                s.flush()
        def isatty(self):
            return any(getattr(s, "isatty", lambda: False)() for s in self.streams)
        def fileno(self):
            return self.streams[0].fileno()
        @property
        def encoding(self):
            return getattr(self.streams[0], "encoding", "utf-8")
        @property
        def errors(self):
            return getattr(self.streams[0], "errors", None)

    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    # Load YAML config
    yaml_config = {}
    if args.config and os.path.exists(args.config):
        yaml_config = load_yaml_config(args.config)
        print(f"Loaded config from {args.config}")

    yaml_overrides = {k: v for k, v in yaml_config.items() if k in RolloutArgs.__dataclass_fields__}
    # These are determined from CLI args, not the yaml.
    yaml_overrides.pop("path_to_save_generated_samples", None)
    # Resolve env mode: explicit --env-mode wins; otherwise env/config.yaml.
    if args.env_mode:
        yaml_overrides["env_mode"] = args.env_mode
    if args.eval_protocol == "webvoyager":
        yaml_overrides["judge_api_model"] = "gpt-4o"
        yaml_overrides["judge_max_attached_imgs"] = 30
    elif args.eval_protocol == "online_mind2web":
        yaml_overrides["judge_api_model"] = "o4-mini"
    elif args.eval_protocol == "deepshop":
        yaml_overrides["judge_api_model"] = "gpt-4o"
        yaml_overrides["judge_max_attached_imgs"] = 30
    eval_args = RolloutArgs(
        hf_checkpoint=args.hf_checkpoint,
        **yaml_overrides,
        path_to_save_generated_samples=args.output if args.save_sample else "",
    )

    init_http_client(eval_args)

    sampling_params = {
        "temperature": eval_args.rollout_temperature,
        "top_p": eval_args.rollout_top_p,
        "top_k": eval_args.rollout_top_k,
        "min_p": eval_args.rollout_min_p,
        "presence_penalty": eval_args.rollout_presence_penalty,
        "repetition_penalty": eval_args.rollout_repetition_penalty,
        "max_new_tokens": eval_args.rollout_max_response_len,
        "stop": eval_args.rollout_stop,
        "stop_token_ids": eval_args.rollout_stop_token_ids,
        "skip_special_tokens": eval_args.rollout_skip_special_tokens,
    }

    # Select tasks to run.
    tasks = load_tasks_from_jsonl(args.task_file, args.task_start, args.task_end, args.shuffle)
    print(f"Loaded {len(tasks)} tasks from {args.task_file} "
          f"(range [{args.task_start}:{args.task_end if args.task_end > 0 else 'end'}])")
    print(f"Max parallel environments: {args.n_parallel}")

    # Save run config (reflects post-override eval_args, not raw yaml).
    config_path = os.path.join(args.output, "eval_config.json")
    with open(config_path, "w") as f:
        eval_args_dict = {k: getattr(eval_args, k) for k in RolloutArgs.__dataclass_fields__}
        merged_cfg = {**yaml_config, **eval_args_dict,
                      **{k: v for k, v in vars(args).items() if v is not None}}
        json.dump(merged_cfg, f, indent=2, default=str)
    print(f"Config saved to {config_path}")

    # Run evaluation
    reward_func = _load_reward_func(args.eval_protocol)
    print(f"Eval protocol: {args.eval_protocol}")
    start_time = datetime.datetime.now()
    results = asyncio.run(evaluate_all(eval_args, tasks, args.n_parallel, sampling_params,
                                       reward_func, args.output))
    elapsed = datetime.datetime.now() - start_time

    summary = get_eval_summary(results)
    print(summary)
    print(f"\nTotal time: {elapsed}")

    summary_path = os.path.join(args.output, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary)
        f.write(f"\nTotal time: {elapsed}\n")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
