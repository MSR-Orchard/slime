#!/usr/bin/env python3
"""Monitor CPU/memory usage of a remote Azure docker service over rolling windows."""

import argparse
import time
import urllib.request
import json
from collections import deque
from datetime import datetime


POLL_INTERVAL = 20  # seconds between samples
PRINT_INTERVAL = 60  # seconds between prints
MAX_WINDOW_SECONDS = 6 * 3600

WINDOWS = [
    ("10min", 10 * 60),
    ("30min", 30 * 60),
    ("1h", 3600),
    ("3h", 3 * 3600),
    ("6h", 6 * 3600),
]


def fetch_resources(ip, api_key, timeout=10):
    """Fetch /resources from remote service. Returns parsed JSON dict or None."""
    url = f"http://{ip}/resources"
    req = urllib.request.Request(url, headers={"X-API-Key": api_key})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None


def compute_sample(data):
    """Extract (cpu_req, cpu_alloc, cpu_pct, mem_req_gib, mem_alloc_gib, mem_pct)."""
    req = data.get("total_requested", {}) or {}
    alloc = data.get("total_allocatable", {}) or {}

    cpu_req = float(req.get("cpu", 0) or 0)
    cpu_alloc = float(alloc.get("cpu", 0) or 0)
    cpu_pct = (cpu_req / cpu_alloc * 100) if cpu_alloc > 0 else 0.0

    mem_req = float(req.get("memory_bytes", 0) or 0) / 1073741824
    mem_alloc = float(alloc.get("memory_bytes", 0) or 0) / 1073741824
    mem_pct = (mem_req / mem_alloc * 100) if mem_alloc > 0 else 0.0

    return cpu_req, cpu_alloc, cpu_pct, mem_req, mem_alloc, mem_pct


def avg_in_window(history, now, window_secs, idx):
    cutoff = now - window_secs
    total, count = 0.0, 0
    for ts, vals in reversed(history):
        if ts < cutoff:
            break
        total += vals[idx]
        count += 1
    return total / count if count else None


def fmt(v, suffix=""):
    return f"{v:6.2f}{suffix}" if v is not None else "   N/A"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", default="", help="Remote service IP/host")
    parser.add_argument("--api-key", default="", help="X-API-Key value")
    parser.add_argument("--poll-interval", type=int, default=POLL_INTERVAL)
    parser.add_argument("--print-interval", type=int, default=PRINT_INTERVAL)
    args = parser.parse_args()

    # history of (timestamp, (cpu_req, cpu_alloc, cpu_pct, mem_req, mem_alloc, mem_pct))
    history = deque()
    last_print = 0

    print(f"Azure Resource Monitor: {args.ip}")
    print(f"Sampling every {args.poll_interval}s, printing every {args.print_interval}s")
    print(f"Rolling windows: {', '.join(name for name, _ in WINDOWS)}")
    print("-" * 100)

    while True:
        now = time.time()
        cutoff = now - MAX_WINDOW_SECONDS

        data = fetch_resources(args.ip, args.api_key)
        if data is not None:
            try:
                sample = compute_sample(data)
                history.append((now, sample))
            except Exception as e:
                print(f"Error parsing response: {e}")

        while history and history[0][0] < cutoff:
            history.popleft()

        if now - last_print >= args.print_interval and history:
            last_print = now
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Latest absolute values
            _, latest = history[-1]
            cpu_req, cpu_alloc, _, mem_req, mem_alloc, _ = latest
            print(f"[{ts}] Current: "
                  f"CPU req={cpu_req:.2f} alloc={cpu_alloc:.2f}  |  "
                  f"Mem req={mem_req:.2f} GiB alloc={mem_alloc:.2f} GiB")

            for win_name, win_secs in WINDOWS:
                cpu_r = avg_in_window(history, now, win_secs, 0)
                cpu_a = avg_in_window(history, now, win_secs, 1)
                mem_r = avg_in_window(history, now, win_secs, 3)
                mem_a = avg_in_window(history, now, win_secs, 4)
                print(f"  [{win_name:>5s}]  "
                      f"CPU req={fmt(cpu_r)} alloc={fmt(cpu_a)}   "
                      f"Mem req={fmt(mem_r, ' GiB')} alloc={fmt(mem_a, ' GiB')}")

            elapsed_min = (history[-1][0] - history[0][0]) / 60
            print(f"  Data: {len(history)} samples over {elapsed_min:.1f} min")
            print()

        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
