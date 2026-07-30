#!/bin/bash
# ==============================================================================
# Evaluate a model on browser tasks (WebVoyager / WebGym).
#
# Runs parallel evaluations using run_evaluate.py, computes rewards via
# LLM judge, and saves per-task results + summary.
#
# Prerequisites:
#   - SGLang server running and accessible (default: localhost:30000)
#   - Evaluation task file(s) in JSONL format
#   - OPENAI_API_KEY and OPENAI_API_BASE set for LLM judge reward
#   - Browser sandbox environment configured
#
# Usage:
#   bash examples/orchard_gui/run_evaluate.sh
# ==============================================================================

trap 'bash "$(dirname "${BASH_SOURCE[0]}")/clean_processes.sh"' EXIT

hf_checkpoints=(
    # HF-format checkpoint dirs (or hub ids); each is evaluated on all task_fns below.
    "examples/orchard_gui/models/Qwen3-VL-4B-Thinking"
)
# Benchmarks to run: task_fns[i] names the task file (examples/orchard_gui/data/<name>.jsonl),
# eval_protocols[i] the matching judge protocol in run_evaluate.py. Keep aligned.
task_fns=(
    'online-mind2web'
    'webvoyager_fara'
    'deepshop'
)
eval_protocols=(
    'online_mind2web'
    'webvoyager'
    'deepshop'
)

for hf_checkpoint in "${hf_checkpoints[@]}"; do
    # Extract model name from checkpoint path (last path component)
    model_name=$(basename "${hf_checkpoint}")

    # Setting up SGLang server and launch hf_checkpoint
    tmux send-keys -t sglang C-c
    sleep 5

    # cd the session to this script's invocation dir so relative checkpoint paths
    # resolve regardless of where the sglang session was originally created.
    tmux send-keys -t sglang "cd '${PWD}'" Enter
    tmux send-keys -t sglang "python -m sglang.launch_server --model-path ${hf_checkpoint} --port 30000 --dp 8 --context-length 36864 --attention-backend flashinfer --mm-attention-backend triton_attn" Enter

    echo "Waiting for SGLang server to be ready..."
    until curl -s http://localhost:30000/health > /dev/null 2>&1; do
        sleep 5
    done
    echo "SGLang server is ready."

    for i in "${!task_fns[@]}"; do
        task_fn="${task_fns[$i]}"
        eval_protocol="${eval_protocols[$i]}"
        task_file="examples/orchard_gui/data/${task_fn}.jsonl"

        echo "============================================"
        echo "HF_CHECKPOINT:   ${hf_checkpoint}"
        echo "TASK_FILE:       ${task_file}"
        echo "EVAL_PROTOCOL:   ${eval_protocol}"
        echo "============================================"
        if [ ! -f "${task_file}" ]; then
            echo "WARNING: Task file not found: ${task_file}, skipping."
            continue
        fi

        # ==========================================================
        # Added loop: Run each experiment 1 times
        # ==========================================================
        for run_id in {1..1}; do
            echo ">>> Starting Run ${run_id}/1 for ${task_fn} <<<"
            
            python examples/orchard_gui/run_evaluate.py \
                --hf-checkpoint "${hf_checkpoint}" \
                --task-file "${task_file}" \
                --eval-protocol "${eval_protocol}" \
                --task-start 0 \
                --task-end -1 \
                --n-parallel 32 \
                --shuffle
                # --save_sample

            echo "Done: ${model_name} / ${task_fn} (Run ${run_id})"

            # Sweep any leftover envs from aborted rollouts. Both are idempotent
            # — run both regardless of mode so switching modes still cleans up.
            python examples/orchard_gui/env/clients/sandbox_env.py --cleanup
            python examples/orchard_gui/env/clients/browser_use_env.py --cleanup
            
        done # End of the 3-run loop

    done

    # Stop SGLang server before launching next checkpoint
    tmux send-keys -t sglang C-c
    sleep 5
done

echo "============================================"
echo "All evaluations complete."
echo "============================================"

# Full node cleanup (sandboxes + sglang server + ray) now that the sweep is done.
bash "$(dirname "${BASH_SOURCE[0]}")/clean_processes.sh"
