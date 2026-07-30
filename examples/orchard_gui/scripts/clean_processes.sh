#!/usr/bin/env bash
# ==============================================================================
# Clean up after a browser RL run so the node is ready for a rerun:
#   1. delete leftover docker sandboxes recorded in .sandboxes/
#   2. tear down the Ray cluster + SGLang engines + the train.py driver
#
# Safe to run repeatedly and when nothing is running (every step is
# best-effort). It deliberately does NOT `pkill -9 python`: that would also
# kill unrelated Python processes on the box. Instead it targets only the
# slime / ray / sglang processes this example starts.
#
# Set (and export) EXTERNAL_RAY=1 to leave a shared Ray cluster untouched.
# ==============================================================================

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
EXTERNAL_RAY="${EXTERNAL_RAY:-0}"

# ── 1. Delete leftover sandboxes ──────────────────────────────────────────────
# Run from the repo root so the script's imports/paths resolve. Bound it with a
# timeout and `|| true` so an unreachable orchestrator can never block (or fail)
# the process teardown below.
( cd "${REPO_ROOT}" && timeout 120 python3 examples/orchard_gui/env/clients/sandbox_env.py --cleanup ) || true

# ── 2. Kill stale processes ───────────────────────────────────────────────────
# Send SIGTERM first and wait a few seconds: a clean exit lets Ray reap its own
# children, which is what avoids the pile of <defunct>/zombie workers a bare
# `ray stop` can leave behind. SIGKILL only whatever is still alive afterward.
# `-f` matches against the full command line.
kill_pattern() {
    local pattern="$1"
    pgrep -f "${pattern}" >/dev/null 2>&1 || return 0   # nothing to do
    pkill -f "${pattern}" 2>/dev/null || true
    for _ in 1 2 3 4 5; do
        pgrep -f "${pattern}" >/dev/null 2>&1 || return 0
        sleep 1
    done
    pkill -9 -f "${pattern}" 2>/dev/null || true
}

# The training driver (`ray job submit -- python3 train.py ...`) and any SGLang
# engine subprocesses. Scoping train.py to a `python` invocation avoids matching
# an editor that happens to have the file open.
kill_pattern 'python[0-9.]*[[:space:]].*train\.py'
kill_pattern 'sglang'

# Ray cluster — skip entirely when pointed at an external/shared cluster.
if [ "${EXTERNAL_RAY}" != "1" ]; then
    ray stop --force >/dev/null 2>&1 || true
    # `ray stop` sometimes leaves the core daemons (and their now-orphaned
    # workers) behind. Killing the daemons by name lets init reap any zombies.
    kill_pattern 'raylet|gcs_server|plasma_store|ray::|redis-server'
fi

echo "clean_processes.sh: cleanup complete."
