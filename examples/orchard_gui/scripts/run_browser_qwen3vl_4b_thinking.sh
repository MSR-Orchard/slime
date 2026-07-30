#!/bin/bash
# ==============================================================================
# Train OpenWebRL-4B-SFT (Qwen3-VL-4B-Thinking based) for Browser Agent tasks.
#
# Hardware: 1 node × 8 GPUs (180GB B200)
# Backend: Megatron (TP=4, SP, dynamic batching)
# Method:  GRPO with LLM-as-a-Judge reward on WebGym tasks (eval: Online-Mind2Web)
#
# Model details:
#   - Dense 4B parameter VL model with <think> reasoning traces
#   - Megatron shards across 4 TP ranks for training (2 DP replicas on 8 GPUs)
#   - 8 parallel SGLang engines for rollout (1 GPU each)
#   - Longer per-turn budget to accommodate thinking tokens
#
# Usage:
#   bash examples/orchard_gui/scripts/run_browser_qwen3vl_4b_thinking.sh
#
# Prerequisites:
#   - base model or SFT checkpoint auto-downloaded from HF 
#     (Qwen/Qwen3-VL-4B-Thinking -> models/Qwen3-VL-4B-Thinking)
#     (OpenWebRL/OpenWebRL-4B-SFT -> models/OpenWebRL-4B-SFT)
#   - Train data: examples/orchard_gui/data/webgym_filtered_20260331_113034.parquet
#   - Eval data:  examples/orchard_gui/data/online-mind2web.parquet
#   - Browser env configured in examples/orchard_gui/env/config.yaml
#   - OPENAI_API_KEY and OPENAI_API_BASE set for LLM judge reward
#   - Megatron-LM installed at /root/Megatron-LM/
# ==============================================================================

set -ex

trap 'bash "$(dirname "${BASH_SOURCE[0]}")/clean_processes.sh"' EXIT

# ==============================================================================
# 1. Configuration
# ==============================================================================

HF_REPO="Qwen/Qwen3-VL-4B-Thinking"
# HF_REPO="OpenWebRL/OpenWebRL-4B-SFT"
HF_CHECKPOINT="examples/orchard_gui/models/${HF_REPO##*/}"

EXPT_NAME="${HF_REPO##*/}_orchard_gui"
SAVE_DIR="examples/orchard_gui/outputs/${EXPT_NAME}"

EXTERNAL_RAY="${SLIME_SCRIPT_EXTERNAL_RAY:-0}"

TRAIN_DATA="examples/orchard_gui/data/webgym_filtered_20260331_113034.parquet"
EVAL_DATA="examples/orchard_gui/data/online-mind2web.parquet"

echo "EXPT_NAME: ${EXPT_NAME}"
echo "HF_CHECKPOINT: ${HF_CHECKPOINT}"
echo "SAVE_DIR: ${SAVE_DIR}"

# ======================================================================================================================
# 2. Prepare: download model & build data
# ======================================================================================================================

mkdir -p "examples/orchard_gui/outputs"
if [ ! -f "${HF_CHECKPOINT}/config.json" ]; then
    hf download "${HF_REPO}" --local-dir "${HF_CHECKPOINT}"
fi

# Task data is shipped pre-built in examples/orchard_gui/data/. To rebuild from raw JSONL, see data/prepare_webgym.py (train)
# and data/prepare_webvoyager.py (eval).
[ -f "${TRAIN_DATA}" ] || { echo "ERROR: ${TRAIN_DATA} not found"; exit 1; }
[ -f "${EVAL_DATA}" ]  || { echo "ERROR: ${EVAL_DATA} not found"; exit 1; }

# ======================================================================================================================
# 3. Environment setup
# ======================================================================================================================

NUM_GPUS=8
export SGLANG_VLM_CACHE_SIZE_MB=0       # browser screenshots overflow default cache
export BROWSER_REWARD_CONCURRENCY="${BROWSER_REWARD_CONCURRENCY:-8}"
export PYTHONBUFFERED=1
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
echo "BROWSER_REWARD_CONCURRENCY: ${BROWSER_REWARD_CONCURRENCY}"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# ======================================================================================================================
# 4. Kill stale processes & start ray
# ======================================================================================================================

pkill -9 sglang || true; sleep 3
if [ "${EXTERNAL_RAY}" != "1" ]; then
    ray stop --force || true; pkill -9 ray || true
fi
pkill -9 slime || true; sleep 3
if [ "${EXTERNAL_RAY}" != "1" ]; then pkill -9 ray || true; fi
pkill -9 slime || true; pkill -9 redis || true

# Qwen3-VL loads HF directly in bridge mode (--megatron-to-hf-mode bridge), which builds
# the full VL model (vision + language). No HF→torch_dist conversion is needed (that's the
# Qwen3.5 raw-mode path, which would yield a language-only model that can't take pixel_values).

if [ "${EXTERNAL_RAY}" != "1" ]; then
    export no_proxy="127.0.0.1,${MASTER_ADDR}"
    ray start --head \
        --node-ip-address "${MASTER_ADDR}" \
        --num-gpus ${NUM_GPUS} \
        --disable-usage-stats \
        --dashboard-host=0.0.0.0 --dashboard-port=8265
fi

# ======================================================================================================================
# 5. Training arguments
# ======================================================================================================================

# ── Checkpoint ────────────────────────────────────────────────────────────────────────────────────────────────────
CKPT_ARGS=(
    --hf-checkpoint "${HF_CHECKPOINT}"     # HF: tokenizer/processor + SGLang engine + bridge weight source
    --save "${SAVE_DIR}"                   # RL output dir
    --save-interval 5
    # --async-save  # disabled 2026-07-29: crash left iter_0000004 un-finalized (no
    #               # latest_checkpointed_iteration.txt) — same loss mode as 9B (2026-07-22)
)

# ── Rollout (browser-specific) ───────────────────────────────────────────────
# Longer response / context budgets than the Instruct variant because the
# Thinking model emits <think>...</think> traces each turn.
ROLLOUT_ARGS=(
    --prompt-data "${TRAIN_DATA}"
    --input-key prompt
    --rollout-shuffle
    # Custom browser generate / reward / config
    --custom-generate-function-path examples.orchard_gui.generate_browser.generate_turn_sample
    --custom-rm-path examples.orchard_gui.reward_browser.reward_func
    --custom-config-path examples/orchard_gui/browser_training_config.yaml
    # --dynamic-sampling-filter-path examples.orchard_gui.filters.check_reward_nonzero_std
    --dynamic-sampling-filter-path examples.orchard_gui.filters.check_reward_nonempty_nonzero_std
    # Adaptive query sampling: weight fresh-query selection by success-rate/staleness/length.
    # NOTE: this re-enables per-rollout all_data accumulation (needed to bucket saturated
    # queries), so a sandbox-outage stuck rollout can re-open the OOM risk — watch memory.
    # Hyper-params
    --num-rollout 100                       # 100 rollout steps × 32 prompt groups = 3200 prompt visits
    --rollout-batch-size 48
    --n-samples-per-prompt 5
    --rollout-max-response-len 2048        # out_seq_length generation cap
    --rollout-max-context-len 32768         # prompt/context cap; separate from the max_new_tokens cap above
    --rollout-temperature 1.0
    --rollout-top-p 1.0
    --rollout-top-k 20
    # global-batch-size controls optimizer steps per rollout:
    #   num_steps = num_trajectories // global_batch_size
    # where num_trajectories = rollout_batch_size * n_samples_per_prompt = 48 * 5 = 240 (all turns of a
    # trajectory share one rollout id, so trajectories — not turns — are counted; see dp_schedule.py).
    # 240 => exactly ONE on-policy update per rollout, which learns slowly.
    # 40 => 6 updates/rollout. Keep it a multiple of n_samples_per_prompt (5) so sampled groups stay intact.
    --global-batch-size 40
)

# ── Eval ─────────────────────────────────────────────────────────────────────
EVAL_ARGS=(
    --eval-interval 5
    --eval-prompt-data onlineM2W "${EVAL_DATA}"
    --n-samples-per-eval-prompt 1
    --eval-temperature 1.0
    --eval-top-p 0.95
    --eval-top-k 20
    --eval-max-response-len 4096
)

# ── GRPO ─────────────────────────────────────────────────────────────────────
GRPO_ARGS=(
    --advantage-estimator grpo
    --kl-loss-coef 0.00
    --kl-loss-type low_var_kl
    --kl-coef 0.00
    --entropy-coef 0.00
    --eps-clip 0.2
    --eps-clip-high 0.28
    --use-rollout-logprobs
    # Reuse each rollout's data for 2 PPO epochs (2x the gradient updates/rollout, more
    # sample-efficient from expensive browser rollouts; mildly off-policy, bounded by eps-clip).
    # Requires --use-rollout-logprobs (fixed IS baseline across epochs).
    --ppo-epochs 2
)

# ── Optimizer ────────────────────────────────────────────────────────────────
OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    # 4B is small — no CPU offload needed on 180GB B200 with Megatron TP=2
    --optimizer-cpu-offload
    --use-precision-aware-optimizer
)

# ── SGLang rollout engine ────────────────────────────────────────────────────
# 4B dense fits on 1 GPU (~8GB bf16) → 8 parallel engines for throughput
# server-concurrency 30 × 8 engines = 240 active trajectories, matching 48 × 5.
SGLANG_ARGS=(
   --sglang-mem-fraction-static 0.5
   --rollout-num-gpus-per-engine 1
#    --use-slime-router
   --sglang-server-concurrency 30
   --sglang-max-running-requests 30
   --sglang-chunked-prefill-size 8192
   # Blackwell (B200): fa4 needs the cutlass DSL (missing in this image) and fa3 is
   # unsupported on Blackwell, so the VLM vision tower uses Triton attention instead.
   # sdpa is the safe (slower) fallback if triton_attn ever fails.
   --sglang-mm-attention-backend triton_attn
)

# ── Megatron backend ─────────────────────────────────────────────────────────
# TP=4 + dynamic batching with a 32768 token cap so any single trajectory
# (bounded by --rollout-max-context-len) fits in one microbatch. 2 DP replicas.
# Full recompute kept for safety; can be dropped for speed if memory allows.
# Auto-resume: continue from the RL checkpoint in SAVE_DIR when one exists (so the
# auto-resume supervisor can relaunch after a stall and pick up where it left off),
# else cold-start from the SFT checkpoint. Resume follows latest_checkpointed_iteration.txt.
if [ -f "${SAVE_DIR}/latest_checkpointed_iteration.txt" ]; then
    LOAD_DIR="${SAVE_DIR}"
    echo "RESUME: --load ${LOAD_DIR} (iter $(cat "${SAVE_DIR}/latest_checkpointed_iteration.txt"))"
else
    LOAD_DIR="${HF_CHECKPOINT}"
    echo "COLD START: --load ${LOAD_DIR}"
fi

BACKEND_ARGS=(
    --train-backend megatron
    --load "${LOAD_DIR}"
    --tensor-model-parallel-size 4
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    # Full activation recompute — essential here: colocate leaves only ~40% of the GPU
    # to training (SGLang reserves --sglang-mem-fraction-static), and a 32k-token VLM
    # micro-batch's activations otherwise OOM. Trades ~30% step speed for the memory.
    # --recompute-granularity full
    # --recompute-method uniform
    # --recompute-num-layers 1
    --micro-batch-size 1                 # ignored when --use-dynamic-batch-size is set
    # Dynamic batching: pack variable-length multi-turn/VLM samples by token budget.
    # 36864 > rollout-max-context-len (32768) so the longest single sample fits one
    # micro-batch (cp_size=1, no sequence splitting). Lower if you OOM.
    --use-dynamic-batch-size
    --max-tokens-per-gpu 32768
    --attention-dropout 0.0
    --hidden-dropout 0.0
    # --accumulate-allreduce-grads-in-fp32
    # --attention-softmax-in-fp32
    # --attention-backend flash
    --attention-backend flash
    --megatron-to-hf-mode bridge
    --moe-token-dispatcher-type alltoall
)

# ── Wandb (optional) ────────────────────────────────────────────────────────
WANDB_ARGS=()
if [ -n "${WANDB_API_KEY}" ]; then
    WANDB_ARGS=(
        --use-wandb
        --wandb-project slime-dev
        --wandb-group "${EXPT_NAME}"   # EXPT_NAME alone: wandb GroupName has a 128-char limit
        --wandb-key "${WANDB_API_KEY}"
        --wandb-random-suffix
        --wandb_always_use_train_step # for clearer mapping
    )
    [ -n "${WANDB_ENTITY}" ] && WANDB_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

# ── Misc ─────────────────────────────────────────────────────────────────────
MISC_ARGS=(
    --actor-num-nodes 1
    --actor-num-gpus-per-node ${NUM_GPUS}
    --colocate
    --enable-adaptive-query-sampling
    # --save-hf "${SAVE_DIR}/hf/rollout_{rollout_id}"
)

# ==============================================================================
# 6. Launch
# ==============================================================================

# Megatron model type: Qwen3-VL-4B-Thinking → qwen3-4B (base dense 4B config)
MEGATRON_MODEL_TYPE="qwen3-4B"
export MODEL_ARGS_ROTARY_BASE=5000000

# Source the megatron model definition to get ${MODEL_ARGS[@]}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${REPO_ROOT}/scripts/models/${MEGATRON_MODEL_TYPE}.sh"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${REPO_ROOT}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"no_proxy\": \"127.0.0.1,${MASTER_ADDR}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"SGLANG_VLM_CACHE_SIZE_MB\": \"0\",
    \"BROWSER_REWARD_CONCURRENCY\": \"${BROWSER_REWARD_CONCURRENCY}\"
  }
}"

export no_proxy="127.0.0.1"

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 train.py \
    ${MODEL_ARGS[@]} \
    ${CKPT_ARGS[@]} \
    ${ROLLOUT_ARGS[@]} \
    ${GRPO_ARGS[@]} \
    ${OPTIMIZER_ARGS[@]} \
    ${SGLANG_ARGS[@]} \
    ${BACKEND_ARGS[@]} \
    ${MISC_ARGS[@]} \
    ${EVAL_ARGS[@]} \
    ${WANDB_ARGS[@]}

bash examples/orchard_gui/scripts/clean_processes.sh