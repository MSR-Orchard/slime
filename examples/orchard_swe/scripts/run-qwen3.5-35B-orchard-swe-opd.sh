#!/bin/bash
# examples/orchard_swe/scripts/run-qwen3.5-35B-orchard-swe-opd.sh
#
# Minimal single-node 8-GPU colocated OPD + SWE-reward run for Qwen3.5-35B-A3B.
# Student training + student rollout share the same 8 GPUs; the teacher SGLang
# server is launched separately (set TEACHER_IP / TEACHER_PORT below).
#
# All the env-toggle / experiment-sweep machinery has been stripped and every
# CLI flag that this branch no longer supports has been removed. Adjust the
# hardcoded defaults inline if you need a different configuration.
#
# Usage:
#   bash examples/orchard_swe/scripts/run-qwen3.5-35B-orchard-swe-opd.sh

ray stop --force
pkill -9 ray
pkill -9 sglang
sleep 3
pkill -9 ray
pkill -9 sglang
rm -f /dev/shm/cuda.shm.* 2>/dev/null
rm -rf /tmp/ray/session_* 2>/dev/null

python examples/orchard_swe/scripts/cleanup_azure_sandboxes.py

set -euo pipefail

# Set WANDB_API_KEY / WANDB_BASE_URL in your shell before launching.
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-}"

# ── Paths ──────────────────────────────────────────────────────────────────
MODEL_NAME=Qwen3.5-35B-A3B
STUDENT_HF=/data/users/shared/models/Qwen/Qwen3.5-35B-A3B
STUDENT_TORCH_DIST=/data/users/shared/models/Qwen/Qwen3.5-35B-A3B_torch_dist_slime-0.3.0
TEACHER_HF=/data/users/shared/models/Qwen/Qwen3.5-397B-A17B-FP8

TEST_DATA_ROOT=/data/users/xxxx/workspace/swe_data/swebench
PROMPT_DATA=/data/users/xxxx/workspace/swe_data/mix/swerebench_v2_scaleswe_combined_0514_v2.jsonl
EVAL_DATA="swe_val ${TEST_DATA_ROOT}/swebench_200_random.jsonl"

STUDENT_NAME=$(basename "${STUDENT_HF}")
TEACHER_NAME=$(basename "${TEACHER_HF}")

SAVE_DIR=/data/users/xxxx/slime_ckpts/${MODEL_NAME}-opd-swe-minimal
LOAD_DIR="${STUDENT_TORCH_DIST}"

export SWE_CONFIG_PATH="examples/orchard_swe/configs/swe_agent_v2_qwen3.5.yaml"
export SLIME_BEST_CKPT_METRIC="eval/swe_val/resolved_count"

# ── Teacher server ─────────────────────────────────────────────────────────
TEACHER_IP="127.0.0.1"
TEACHER_PORT=30002

# ── SWE / reward runtime settings ──────────────────────────────────────────
export SWE_MAX_ALL_TOKENS=65536
export FLASHINFER_USE_CUDA_NORM=1

export SWE_REWARD_MODE="simple"
export SWE_UNRESOLVED_REWARD=0.0
export SWE_TIMEOUT_CREATE_ENV=480
export SWE_TIMEOUT_LLM_INFERENCE=60
export SWE_TIMEOUT_GET_OBSERVATION=90
export SWE_TIMEOUT_STOP_ENV=60
export SWE_TIMEOUT_REWARD_EXECUTE=600
export SWE_TIMEOUT_REWARD_TOTAL=600
export SWE_MAX_ENV_RETRIES=1
export SWE_ENV_RETRY_WAIT=10
export SWE_MAX_CREATE_ENV_RETRIES=1
export SWE_CREATE_ENV_RETRY_WAIT=10
export SWE_ENV_CREATE_JITTER_MAX=5

# ── Batch-size config ──────────────────────────────────────────────────────
NUM_NODES=1
ROLL_BS=32
NPROMPT_PER_INSTANCE=1
TRAIN_BS=16
OVER_SAMPLING_BATCH_SIZE=32

# ── OPD training mode ──────────────────────────────────────────────────────
# OPD_ONLY=1 -> OPD only: zero out the GRPO task-reward advantage so only the OPD
#              KL distillation signal trains the student (--zero-train-adv-for-opd).
# OPD_ONLY=0 -> OPD + GRPO: keep the SWE task-reward advantage AND the OPD KL
#              signal. Override via env var, e.g. OPD_ONLY=1 bash <script>.
OPD_ONLY="${OPD_ONLY:-0}"
if [[ "${OPD_ONLY}" == "1" ]]; then
    zero_adv_flag="--zero-train-adv-for-opd"
else
    zero_adv_flag=""
fi
# OPD KL coefficient (--opd-kl-coef). Override via env var, e.g. OPD_KL_COEF=0.5.
OPD_KL_COEF="${OPD_KL_COEF:-1.0}"
# OPD loss surrogate (--opd-reverse-kl-loss-type). One of: k1, k2, k3, js_skew.
# Override via env var, e.g. OPD_LOSS_TYPE=k3 bash <script>.
OPD_LOSS_TYPE="${OPD_LOSS_TYPE:-js_skew}"
# Teacher mixture weight lambda for the js_skew loss (--opd-js-mixture-weight),
# in (0, 1]. Only used when OPD_LOSS_TYPE=js_skew. Override via env var.
OPD_JS_MIXTURE_WEIGHT="${OPD_JS_MIXTURE_WEIGHT:-0.5}"
# ── OPD config (teacher endpoint + cross-vocab alignment) ──────────────────
RUNTIME_OPD_CONFIG="/data/users/xxxx/slime_runtime_configs/opd_config_minimal_$$.yaml"
mkdir -p "$(dirname "${RUNTIME_OPD_CONFIG}")"
cat > "${RUNTIME_OPD_CONFIG}" <<EOF
teacher:
  url: "http://${TEACHER_IP}:${TEACHER_PORT}/generate"
  tokenizer_path: "${TEACHER_HF}"

timeout: 600
teacher_truncation_side: suffix
teacher_max_len: 75000
teacher_topk: 1
EOF

# ── Argument groups ────────────────────────────────────────────────────────
WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-opd-debug
   --wandb-group "${MODEL_NAME}-opd-swe-minimal"
   --wandb-key "${WANDB_API_KEY}"
   --disable-wandb-random-suffix
)

EVAL_ARGS=(
   --skip-eval-before-train
   --eval-interval 5
   --eval-prompt-data ${EVAL_DATA}
   --n-samples-per-eval-prompt 1
   --eval-max-concurrency 64
   --eval-max-response-len 8192
   --eval-top-p 0.95
   --eval-temperature 1.0
   --eval-task-timeout 1500
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-ep-size 2
   --sglang-moe-runner-backend triton
   --sglang-attention-backend trtllm_mha
   --sglang-mamba-scheduler-strategy extra_buffer
   --sglang-mem-fraction-static 0.7
   --sglang-server-concurrency 256
   --sglang-max-running-requests 256
   --sglang-chunked-prefill-size 4096
   --sglang-data-parallel-size 2
   --sglang-enable-dp-attention
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 ${TRAIN_BS})
)

ROLLOUT_ARGS=(
   --rollout-shuffle
   --over-sampling-batch-size ${OVER_SAMPLING_BATCH_SIZE}
   --prompt-data "${PROMPT_DATA}"
   --multimodal-keys '{}'
   --input-key problem_statement
   --label-key patch
   --num-rollout 300
   --rollout-batch-size ${ROLL_BS}
   --n-samples-per-prompt ${NPROMPT_PER_INSTANCE}
   --rollout-max-response-len 4096
   --rollout-temperature 1.0
   --rollout-top-p 0.95
)

OPT_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.98
   --use-precision-aware-optimizer
   --calculate-per-token-loss
)

REWARD_ARGS=(
   --custom-generate-function-path examples.orchard_swe.swe_generate_v2.generate
   --custom-rm-path examples.orchard_swe.swe_opd_reward.reward_func
   --custom-reward-post-process-path examples.orchard_swe.swe_opd_reward.post_process_rewards
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted
   --rm-url "http://${TEACHER_IP}:${TEACHER_PORT}/generate"
   --opd-config "${RUNTIME_OPD_CONFIG}"
)

# ── Model arch ─────────────────────────────────────────────────────────────
source "scripts/models/${MODEL_NAME/Q/q}.sh"

# ── Ray (single node) ──────────────────────────────────────────────────────
export MASTER_ADDR="$(hostname -I | awk '{print $1}')"
export no_proxy="127.0.0.1,localhost,${MASTER_ADDR}"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

RAY_OBJECT_STORE_MEMORY=$(( 32 * 1024 * 1024 * 1024 ))
ray start --head \
    --node-ip-address "${MASTER_ADDR}" \
    --num-gpus 8 \
    --object-store-memory ${RAY_OBJECT_STORE_MEMORY} \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265

sleep 5
mkdir -p "${SAVE_DIR}"
cp "$0" "${SAVE_DIR}/run_config.sh"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"FLASHINFER_USE_CUDA_NORM\": \"${FLASHINFER_USE_CUDA_NORM}\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"no_proxy\": \"${no_proxy}\",
    \"WANDB_API_KEY\": \"${WANDB_API_KEY}\",
    \"WANDB_BASE_URL\": \"${WANDB_BASE_URL}\",
    \"SWE_REWARD_MODE\": \"${SWE_REWARD_MODE}\",
    \"SWE_CONFIG_PATH\": \"${SWE_CONFIG_PATH}\",
    \"SWE_TIMEOUT_CREATE_ENV\": \"${SWE_TIMEOUT_CREATE_ENV}\",
    \"SWE_TIMEOUT_LLM_INFERENCE\": \"${SWE_TIMEOUT_LLM_INFERENCE}\",
    \"SWE_TIMEOUT_GET_OBSERVATION\": \"${SWE_TIMEOUT_GET_OBSERVATION}\",
    \"SWE_TIMEOUT_STOP_ENV\": \"${SWE_TIMEOUT_STOP_ENV}\",
    \"SWE_TIMEOUT_REWARD_EXECUTE\": \"${SWE_TIMEOUT_REWARD_EXECUTE}\",
    \"SWE_TIMEOUT_REWARD_TOTAL\": \"${SWE_TIMEOUT_REWARD_TOTAL}\",
    \"SWE_MAX_ENV_RETRIES\": \"${SWE_MAX_ENV_RETRIES}\",
    \"SWE_ENV_RETRY_WAIT\": \"${SWE_ENV_RETRY_WAIT}\",
    \"SWE_MAX_CREATE_ENV_RETRIES\": \"${SWE_MAX_CREATE_ENV_RETRIES}\",
    \"SWE_CREATE_ENV_RETRY_WAIT\": \"${SWE_CREATE_ENV_RETRY_WAIT}\",
    \"SWE_ENV_CREATE_JITTER_MAX\": \"${SWE_ENV_CREATE_JITTER_MAX}\"
  }
}"

# ── Train (8 GPUs colocated: train + rollout share GPUs) ───────────────────
ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 train.py \
    --actor-num-nodes ${NUM_NODES} \
    --actor-num-gpus-per-node 8 \
    --colocate \
    --distributed-timeout-minutes 60 \
    \
    "${MODEL_ARGS[@]}" \
    \
    --hf-checkpoint "${STUDENT_HF}" \
    --ref-load    "${STUDENT_TORCH_DIST}" \
    --load        "${LOAD_DIR}" \
    --save        "${SAVE_DIR}" \
    --save-interval 5 \
    \
    --global-batch-size ${TRAIN_BS} \
    --update-weight-buffer-size $(( 128 * 1024 * 1024 )) \
    --balance-data \
    "${ROLLOUT_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${OPT_ARGS[@]}" \
    "${REWARD_ARGS[@]}" \
    \
    --advantage-estimator grpo \
    --use-opd \
    --opd-type sglang \
    --opd-mode loss \
    --opd-kl-coef ${OPD_KL_COEF} \
    ${zero_adv_flag} \
    --opd-reverse-kl-loss-type ${OPD_LOSS_TYPE} \
    --opd-js-mixture-weight ${OPD_JS_MIXTURE_WEIGHT} \
    --use-kl-loss \
    --kl-loss-coef 0.0 \
    --kl-loss-type low_var_kl \
    --entropy-coef 0.0 \
    --eps-clip 0.2 \
    --eps-clip-high 0.28 \
    \
    --tensor-model-parallel-size 2 \
    --sequence-parallel \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 4 \
    --expert-model-parallel-size 8 \
    --expert-tensor-parallel-size 1 \
    \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    \
    --use-dynamic-batch-size \
    --max-tokens-per-gpu 16384 \
    \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --accumulate-allreduce-grads-in-fp32 \
    --attention-softmax-in-fp32 \
    --moe-token-dispatcher-type flex \
    --moe-enable-deepep \
    --attention-backend flash

# ── Cleanup ────────────────────────────────────────────────────────────────
rm -f "${RUNTIME_OPD_CONFIG}"
pkill -9 sglang || true
sleep 3
ray stop --force
pkill -9 ray    || true
pkill -9 python || true
