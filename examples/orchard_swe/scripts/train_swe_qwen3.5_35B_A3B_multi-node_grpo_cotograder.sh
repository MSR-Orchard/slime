#!/bin/bash
# Multi-node training script for SWE-bench with slime (Qwen3.5-35B-A3B, 2 nodes x 8 GPUs)
# Uses the GRPO advantage estimator with an LLM-as-judge process reward (PRM) mixed
# into the outcome reward. Set PRM_ALPHA=0 to fall back to the outcome reward only.

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONUNBUFFERED=1
export SAVE_MODEL_ROOT=/data/users/xxxx/ckpt
export TEST_DATA_ROOT=/data/users/xxxx/workspace/swe_data/swebench
export MODEL_ARGS_ROTARY_BASE=10000000
# rollout trajectories will be written into this dir
export SWE_TRAJECTORY_DIR=/data/users/xxxx/workspace/swe_trajectories_qwen3.5

export PYTHONPATH=/root/Megatron-LM/:${PYTHONPATH}

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "/root/slime/scripts/models/qwen3.5-35B-A3B.sh"

# SWE-specific settings (passed via environment variables)
# export SWE_REWARD_MODE="simple"
export SWE_REWARD_MODE="simple_plus_llm_as_judges"
# Eval rollouts override the above with this mode (no PRM API call).
# Set to the same value as SWE_REWARD_MODE if you actually want PRM during eval.
export SWE_EVAL_REWARD_MODE="simple"
export SWE_UNRESOLVED_REWARD="-1.0"
#export SWE_UNRESOLVED_REWARD="-0.5"
export SWE_CONFIG_PATH="examples/orchard_swe/configs/swe_agent_v2_qwen3.5.yaml"
export SWE_TIMEOUT_CREATE_ENV=480
export SWE_TIMEOUT_LLM_INFERENCE=60
export SWE_TIMEOUT_GET_OBSERVATION=90
export SWE_TIMEOUT_STOP_ENV=60
export SWE_TIMEOUT_REWARD_EXECUTE=360
export SWE_TIMEOUT_REWARD_TOTAL=420
export SWE_MAX_ENV_RETRIES=2
export SWE_ENV_RETRY_WAIT=10
export SWE_MAX_CREATE_ENV_RETRIES=1
export SWE_CREATE_ENV_RETRY_WAIT=10
export SWE_ENV_CREATE_JITTER_MAX=5

# ===== LLM-as-Judge process reward (PRM) — uses GitHub Copilot API =====
# Set PRM_ALPHA=0 to short-circuit and behave exactly like simple_truncated_zero
# (no judge API call, no extra cost).
export PRM_ALPHA=0.5
export PRM_API_TYPE=gh
export PRM_API_BASE_URL=https://api.githubcopilot.com
export PRM_MODEL="claude-sonnet-4.6"
export PRM_ENDPOINT=chat                  # "chat" | "responses" (gpt-5.x reasoning)
# export PRM_REASONING_EFFORT=medium      # only with PRM_ENDPOINT=responses
export PRM_TEMPERATURE=0.6
export PRM_MAX_TOKENS=1024
export PRM_OBS_MODE=compact               # "full" | "compact" | "none"
export PRM_OBS_MAXCHARS=1200
export PRM_TIMEOUT=90                     # per-call timeout inside score_sample
export PRM_TIMEOUT_TOTAL=120              # outer wait_for in reward_func (< SWE_TIMEOUT_REWARD_TOTAL)

# Capture gh token on the launcher node; Ray workers won't have `gh` installed.
if [ -z "${PRM_API_KEY:-}" ]; then
    if command -v gh >/dev/null 2>&1; then
        export PRM_API_KEY="$(gh auth token 2>/dev/null || true)"
    fi
fi
if [ "${PRM_ALPHA}" != "0" ] && [ "${PRM_ALPHA}" != "0.0" ] && [ -z "${PRM_API_KEY}" ]; then
    echo "ERROR: PRM_ALPHA != 0 but PRM_API_KEY is empty and 'gh auth token' unavailable." >&2
    exit 1
fi

# Key hyperparameters (used in argument arrays and wandb naming)
MODEL_NAME="qwen3.5-35b-a3b"
OVER_SAMPLING_BATCH_SIZE=8
ROLLOUT_BATCH_SIZE=8
# N_SAMPLES (n-samples-per-prompt): training group size per prompt
N_SAMPLES=8
GLOBAL_BATCH_SIZE=32    # <= ROLLOUT_BATCH_SIZE * N_SAMPLES
ADVANTAGE_ESTIMATOR="grpo"
LR="1e-6"
NUM_NODES=1
PPO_EPOCHS=1

RANDOM_SUFFIX="$(cat /dev/urandom | LC_ALL=C tr -dc 'a-z0-9' | head -c 6)"
export WANDB_RANDOM_SUFFIX="qwen3.5-35b-a3b_grpo_prm_${RANDOM_SUFFIX}"
WANDB_GROUP_NAME="${MODEL_NAME}_prm_pe${PPO_EPOCHS}_r${ROLLOUT_BATCH_SIZE}_n${N_SAMPLES}_g${GLOBAL_BATCH_SIZE}_${ADVANTAGE_ESTIMATOR}_lr${LR}_${NUM_NODES}nodes_${WANDB_RANDOM_SUFFIX}"


CKPT_SUBDIR="prm_pe${PPO_EPOCHS}_r${ROLLOUT_BATCH_SIZE}_n${N_SAMPLES}_g${GLOBAL_BATCH_SIZE}_${ADVANTAGE_ESTIMATOR}_lr${LR}_${NUM_NODES}nodes_${RANDOM_SUFFIX}"

# Point --hf-checkpoint at the (optionally SFT-ed) HF model and --ref-load at its
# torch_dist conversion produced by the slime checkpoint converter.
CKPT_ARGS=(
   --hf-checkpoint /data/users/shared/models/Qwen/Qwen3.5-35B-A3B
   --ref-load /data/users/shared/models/Qwen/Qwen3.5-35B-A3B_torch_dist_slime-0.3.0
   --load ${SAVE_MODEL_ROOT}/Qwen3.5-35B-A3B/${CKPT_SUBDIR}/
   --save ${SAVE_MODEL_ROOT}/Qwen3.5-35B-A3B/${CKPT_SUBDIR}/
   --save-interval 100
)

ROLLOUT_ARGS=(
   --prompt-data /data/users/xxxx/workspace/swe_data/mix/swerebench_v2_scaleswe_combined_0514_v4.jsonl
   --input-key problem_statement
   --label-key patch
   # qwen3.5 ships a multimodal processor, so prompts must be in conversation (list) form.
   # An empty multimodal map forces list form WITHOUT applying a chat template (avoids double-templating).
   --multimodal-keys '{}'
   --rollout-shuffle

   --custom-generate-function-path examples.orchard_swe.swe_generate_v2.generate
   --custom-rm-path examples.orchard_swe.swe_reward.reward_func
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted_nonzero_std_and_pos_reward
   #--calculate-per-token-loss

   --num-rollout 300
   --over-sampling-batch-size ${OVER_SAMPLING_BATCH_SIZE}
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
   --n-samples-per-prompt ${N_SAMPLES}
   --rollout-max-response-len 6122
   --rollout-temperature 0.95
   --rollout-top-p 0.95

   --seed 1111
   --rollout-seed 1111

   # # qwen3.5 stop token ids (im_end / endoftext in the 248320 vocab)
   # --rollout-stop-token-ids 248046 248044

   --global-batch-size ${GLOBAL_BATCH_SIZE}
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 5
   #--skip-eval-before-train
   --eval-prompt-data swe_val ${TEST_DATA_ROOT}/swebench_200_random.jsonl
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 8192
   --eval-top-p 0.95
   --eval-temperature 0.95
   --eval-task-timeout 1500
)

PERF_ARGS=(
   # Adjusted parallelism for multi-node (2 nodes x 8 GPUs = 16 GPUs)
   # Actor/training TP. Independent of SGLang serving TP (the GDN weight-sync bug was
   # on the serving side, fixed via DP-attention in SGLANG_ARGS). TP=1 OOMs the 35B
   # training model, so keep TP=2 (TP2 x CP4 x DP2 = 16 GPUs).
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 4
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384
)

GRPO_ARGS=(
   --advantage-estimator ${ADVANTAGE_ESTIMATOR}
   # PPO epochs: train each rollout batch this many times (off-policy passes
   # corrected by eps-clip). 1 = on-policy single pass. LR schedule still
   # advances once per rollout, not per epoch.
   --ppo-epochs ${PPO_EPOCHS}
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 1e-4
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr ${LR}
   --lr-decay-style cosine
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   --use-wandb
   --wandb-host ${WANDB_BASE_URL}
   --wandb-project orchard-swe-agent
   --wandb-group ${WANDB_GROUP_NAME}
   --disable-wandb-random-suffix
   --wandb-key ${WANDB_API_KEY}
)

SGLANG_ARGS=(
   --sglang-moe-runner-backend triton
   # Engine spans 2 GPUs. We use DP-attention (+EP) instead of TP for serving so the
   # hybrid GDN params (conv1d / A_log / dt_bias / in_proj_*) are REPLICATED, not
   # TP-sharded. SGLang 0.5.12.post1's update_weights_from_tensor mis-shards those GDN
   # params under engine-TP>1 -> garbage rollout right after the first weight sync.
   # DP-attention sidesteps that buggy path; experts are EP-sharded for memory.
   --rollout-num-gpus-per-engine 2
   --sglang-enable-dp-attention
   --sglang-data-parallel-size 2
   --sglang-expert-parallel-size 2
   --sglang-server-concurrency 64
   --sglang-max-running-requests 64
   #--sglang-chunked-prefill-size 8192
   --sglang-chunked-prefill-size 16384
   --sglang-mem-fraction-static 0.7

   --sglang-mamba-scheduler-strategy extra_buffer

   --sglang-tool-call-parser qwen3_coder
   --sglang-reasoning-parser qwen3
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash

   # qwen3.5 specific: flex MoE dispatcher + DeepEP (overrides alltoall from model config)
   --moe-token-dispatcher-type flex
   --moe-enable-deepep
)



# === Multi-node Ray cluster setup ===
# Use MASTER_ADDR from environment, or VC_MASTER_HOSTS (common in Azure ML), or fallback to localhost
export MASTER_ADDR=${MASTER_ADDR:-${VC_MASTER_HOSTS:-"127.0.0.1"}}
export no_proxy="127.0.0.1,${MASTER_ADDR}"

# Clean up any previous processes on all nodes
pkill -9 sglang || true
sleep 3
ray stop --force || true
pkill -9 ray || true
pkill -9 python || true
sleep 3
pkill -9 ray || true
pkill -9 python || true

# Start Ray head node
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Wait for GCS to be fully ready before starting workers
echo "Waiting for Ray GCS to become available on ${MASTER_ADDR}:6379..."
for i in $(seq 1 30); do
  if python3 -c "import socket; s=socket.create_connection(('${MASTER_ADDR}', 6379), timeout=2); s.close()" 2>/dev/null; then
    echo "GCS is reachable after ${i} seconds."
    break
  fi
  if [ "$i" -eq 30 ]; then
    echo "ERROR: GCS on ${MASTER_ADDR}:6379 not reachable after 30s. Check firewall/network."
    exit 1
  fi
  sleep 1
done
sleep 5

# Start Ray workers on other nodes (reads IPs from ~/hostfile)
for WORKER_IP in $(awk '{print $1}' ~/hostfile); do
  if [[ "$WORKER_IP" == "$MASTER_ADDR" ]]; then
    continue
  fi
  echo "Starting Ray worker on ${WORKER_IP}"
  ssh root@"${WORKER_IP}" \
    "pkill -9 sglang ; ray stop --force ; pkill -9 python ; sleep 2 ; ray start --address=${MASTER_ADDR}:6379 --num-gpus 8 --node-ip-address ${WORKER_IP} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265" &
done
wait

# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"no_proxy\": \"${no_proxy}\",
    \"SWE_REWARD_MODE\": \"${SWE_REWARD_MODE}\",
    \"SWE_EVAL_REWARD_MODE\": \"${SWE_EVAL_REWARD_MODE}\",
    \"SWE_UNRESOLVED_REWARD\": \"${SWE_UNRESOLVED_REWARD}\",
    \"SWE_MAX_STEPS\": \"${SWE_MAX_STEPS}\",
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
    \"SWE_ENV_CREATE_JITTER_MAX\": \"${SWE_ENV_CREATE_JITTER_MAX}\",
    \"PRM_ALPHA\": \"${PRM_ALPHA}\",
    \"PRM_API_TYPE\": \"${PRM_API_TYPE}\",
    \"PRM_API_BASE_URL\": \"${PRM_API_BASE_URL}\",
    \"PRM_API_KEY\": \"${PRM_API_KEY}\",
    \"PRM_MODEL\": \"${PRM_MODEL}\",
    \"PRM_ENDPOINT\": \"${PRM_ENDPOINT}\",
    \"PRM_TEMPERATURE\": \"${PRM_TEMPERATURE}\",
    \"PRM_MAX_TOKENS\": \"${PRM_MAX_TOKENS}\",
    \"PRM_OBS_MODE\": \"${PRM_OBS_MODE}\",
    \"PRM_OBS_MAXCHARS\": \"${PRM_OBS_MAXCHARS}\",
    \"PRM_TIMEOUT\": \"${PRM_TIMEOUT}\",
    \"PRM_TIMEOUT_TOTAL\": \"${PRM_TIMEOUT_TOTAL}\",
    \"WANDB_API_KEY\": \"${WANDB_API_KEY}\",
    \"WANDB_BASE_URL\": \"${WANDB_BASE_URL}\"
  }
}"

#--debug-rollout-only \
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -u train.py \
   --actor-num-nodes ${NUM_NODES} \
   --actor-num-gpus-per-node 8 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}

echo "Multi-node training complete!"
