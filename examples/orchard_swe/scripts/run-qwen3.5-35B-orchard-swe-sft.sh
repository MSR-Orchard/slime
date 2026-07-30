#!/bin/bash
# Multi-node SFT script for SWE agent trajectories with slime (Qwen3.5-35B-A3B).
set -ex

date=$(date +%Y-%m-%d)
echo "Current date and time: ${date}"

SCHEMES=(   
   all_resolved_full_ohb_64k_credit_sft_fixed
)

DATA_PARENT=/data/users/xxxx/data/agentic_rollouts_training/
DATA_DIR=${DATA_PARENT}/merged-0322

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

POLICY_MODEL=Qwen3.5-35B-A3B

REMOVE_REASONING_FLAG=false

SAVE_PARENT=/data/users/xxxx/results/saves
SAVE_DIR=${SAVE_PARENT}/${POLICY_MODEL}-sft-bs512-lr1e-5-epoch5

NUM_NODES=4

for SCHEME in "${SCHEMES[@]}"; do
   echo "=========================================="
   echo "Training scheme: ${SCHEME}"
   echo "=========================================="

   # Clean up previous processes
   pkill -9 sglang || true
   sleep 3
   ray stop --force || true
   pkill -9 ray || true
   pkill -9 python || true
   sleep 3
   pkill -9 ray || true
   pkill -9 python || true

   export PYTHONBUFFERED=16
   # Set WANDB_API_KEY / WANDB_BASE_URL in your shell before launching.
   export WANDB_API_KEY="${WANDB_API_KEY:-}"
   export WANDB_BASE_URL="${WANDB_BASE_URL:-}"

   NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
   if [ "$NVLINK_COUNT" -gt 0 ]; then
       HAS_NVLINK=1
   else
       HAS_NVLINK=0
   fi
   
   source "/root/slime/scripts/models/qwen3.5-35B-A3B.sh"

   CKPT_ARGS=(
      --hf-checkpoint /data/users/shared/models/Qwen/${POLICY_MODEL}
      --ref-load /data/users/shared/models/Qwen/${POLICY_MODEL}_torch_dist_slime-0.3.0
      --load "${SAVE_DIR}/${SCHEME}"
      --save "${SAVE_DIR}/${SCHEME}"
      --save-interval 10000
   )

   SFT_ARGS=(
      --rollout-function-path slime.rollout.sft_rollout.generate_rollout
      --prompt-data "${DATA_DIR}/${SCHEME}.jsonl"
      --input-key messages
      --tool-key tools
      --rollout-max-prompt-len 65536
      --rollout-shuffle
      --num-epoch 5
      --rollout-batch-size 512
      --global-batch-size 512

      --loss-type sft_loss
      --loss-mask-type qwen3_5
      --calculate-per-token-loss
      --disable-compute-advantages-and-returns
      --debug-train-only
      $([ "$REMOVE_REASONING_FLAG" = true ] && echo "--remove-reasoning-content" || true)
   )

   PERF_ARGS=(
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
      --max-tokens-per-gpu 32768
   )

   OPTIMIZER_ARGS=(
      --optimizer adam
      --lr 1e-5
      --lr-decay-style cosine
      --min-lr 1e-6
      --lr-warmup-fraction 0.1
      --weight-decay 0.1
      --adam-beta1 0.9
      --adam-beta2 0.95

      --use-distributed-optimizer
      --optimizer-cpu-offload
      --overlap-cpu-optimizer-d2h-h2d
      --use-precision-aware-optimizer
   )

   WANDB_ARGS=(
      --use-wandb
      --wandb-project slime-sft-swe
      --wandb-group "${POLICY_MODEL}-$(basename ${DATA_DIR})-${SCHEME}-woreason-${REMOVE_REASONING_FLAG}"
      --wandb-key "${WANDB_API_KEY}"
   )

   MISC_ARGS=(
      --attention-dropout 0.0
      --hidden-dropout 0.0
      --accumulate-allreduce-grads-in-fp32
      --attention-softmax-in-fp32
      --attention-backend flash

      --moe-token-dispatcher-type flex
      --moe-enable-deepep
   )

   export MASTER_ADDR=${MASTER_ADDR:-${VC_MASTER_HOSTS:-"127.0.0.1"}}
   export no_proxy="127.0.0.1,${MASTER_ADDR}"
   ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

   # Start Ray workers on other nodes
   for WORKER_IP in $(awk '{print $1}' ~/hostfile); do
     if [[ "$WORKER_IP" == "$MASTER_ADDR" ]]; then
       continue
     fi
     echo "Starting Ray worker on ${WORKER_IP}"
     ssh root@"${WORKER_IP}" \
       "pkill -9 sglang ; ray stop --force ; pkill -9 python ; ray start --address=${MASTER_ADDR}:6379 --num-gpus 8 --node-ip-address ${WORKER_IP} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265" &
   done
   wait

   RUNTIME_ENV_JSON="{
     \"env_vars\": {
       \"PYTHONPATH\": \"/root/Megatron-LM/\",
       \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
       \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
       \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\",
       \"no_proxy\": \"${no_proxy}\",
       \"MASTER_ADDR\": \"${MASTER_ADDR}\",
       \"WANDB_API_KEY\": \"${WANDB_API_KEY}\",
       \"WANDB_BASE_URL\": \"${WANDB_BASE_URL}\"
     }
   }"

   ray job submit --address="http://127.0.0.1:8265" \
      --runtime-env-json="${RUNTIME_ENV_JSON}" \
      -- python3 train_async.py \
      --actor-num-nodes ${NUM_NODES} \
      --actor-num-gpus-per-node 8 \
      ${MODEL_ARGS[@]} \
      ${CKPT_ARGS[@]} \
      ${SFT_ARGS[@]} \
      ${OPTIMIZER_ARGS[@]} \
      ${WANDB_ARGS[@]} \
      ${PERF_ARGS[@]} \
      ${MISC_ARGS[@]}

   echo "Finished training: ${SCHEME}"
done

echo "All schemes completed."
