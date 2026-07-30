# Mini-SWE-Agent Integration for Slime

This package integrates [mini-swe-agent](https://github.com/logic-star-ai/mini-swe-agent) with slime's RL
training framework, enabling RL training on SWE-bench style coding tasks.

It allows you to:

- Train LLMs on SWE-bench tasks using RL (GRPO / PPO / dynamic sampling).
- Use mini-swe-agent's environment for code execution inside sandboxes.
- Correctly distinguish **actions** (trained on) from **observations** (masked out) via loss masking.

> Placeholders used throughout this document: replace `xxxx` with your own user name and `xx.xx.xx.xx` with
> your own endpoint IP.

## Table of contents

1. [Directory layout](#directory-layout)
2. [Environment setup](#1-environment-setup)
3. [Model conversion](#2-model-conversion)
4. [Prepare data](#3-prepare-data)
5. [Configure paths](#4-configure-paths)
6. [Launch training](#5-launch-training)
7. [Convert checkpoints back to HF](#6-convert-checkpoints-back-to-hf)
8. [Key hyperparameters](#key-hyperparameters)
9. [Analysis tools](#analysis-tools)

## Directory layout

| Path | Purpose |
| --- | --- |
| `scripts/` | Data preparation, training launchers, checkpoint conversion, sandbox cleanup |
| `configs/` | mini-swe-agent prompt/agent configs (e.g. `swe_agent_v2_qwen3.5.yaml`) |
| `b200_config/` | Kubernetes job specs for the B200 cluster (1-node / 2-node) plus host generation |
| `analysis/` | Reward, timing and CPU-usage analysis utilities |
| `patch/` | Patches applied to third-party repos (e.g. SWE-bench log parsers) |
| `swe_*.py` | Rollout generation, reward, PRM, filters and agent wrappers used by slime |
| `README_OPD.md` | On-policy distillation variant of this pipeline |

## 1. Environment setup

The base image is `slimerl/slime:v0.3.0` (mirrored as `mirror.gcr.io/slimerl/slime:v0.3.0`). It already
contains SGLang, Megatron-LM, PyTorch and CUDA, so only the SWE-specific extras are missing.

> **Recommended practice: keep the install commands in the job spec, not in a custom image.**
> The pod templates in `b200_config/*.yaml` install everything at container startup (see the `args:` block of
> each task). `slime`, `SWE-bench-fork` and `orchard_env` are all installed with `pip install -e .` from a
> shared PVC (`/data/...`), so code edits take effect on the next job without rebuilding or re-pushing an
> image.

### 1.1 Prepare source repos on the shared volume (once, outside the job)

```bash
cd /data/users/xxxx/workspace/
git clone https://github.com/SWE-rebench/SWE-bench-fork.git SWE-bench-fork
cd SWE-bench-fork
git checkout 71aaff544c63b57943056b05a43271afc475e7b7
cp /data/users/xxxx/workspace/orchard/trainer/slime/examples/orchard_swe/patch/swebench/harness/log_parsers/python.py \
   /data/users/xxxx/workspace/SWE-bench-fork/swebench/harness/log_parsers/python.py
```

### 1.2 Container bootstrap

The following commands turn the base image into a ready-to-train environment. They are already part of the
job YAML, and are listed here so they can also be run manually when debugging in an interactive pod:

```bash
# 1. System packages
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y screen jq
apt-get install -y --no-install-recommends openssh-server openssh-client ca-certificates \
  ibverbs-utils rdmacm-utils perftest infiniband-diags
apt-get install -y pssh          # multi-node jobs only

# 2. Point /root/slime at this repo on the shared volume, then install it without touching pinned deps
mv /root/slime /root/slime_backup
ln -s /data/users/xxxx/workspace/orchard/trainer/slime /root/slime
cd /root/slime && pip install -e . --no-deps

# 3. Python dependencies for the SWE agent pipeline
pip install datasets==4.4.1 pyyaml==6.0.3 jinja2==3.1.6
pip install mini-swe-agent==1.17.5
pip install openai==2.6.1
pip uninstall -y litellm

# 4. Blackwell (B200) kernel fix required by the v0.3.0 image
pip install --force-reinstall --no-deps "nvidia-cutlass-dsl==4.5.1" "nvidia-cutlass-dsl-libs-base==4.5.1"

# 5. Editable installs of the SWE harness and the sandbox client
cd /data/users/xxxx/workspace/SWE-bench-fork/ && pip install -e .
cd /data/users/xxxx/workspace/orchard/orchard_env/ && pip install -e .
```

Notes:

- `--no-deps` on the slime install is important: the image already ships a consistent
  SGLang/Megatron/PyTorch stack, and resolving slime's full dependency tree can downgrade it.
- Versions are pinned deliberately; changing `mini-swe-agent`, `datasets` or `openai` can break trajectory
  parsing and loss masking.
- For multi-node jobs the exact same block runs on every task (master and workers), plus `pssh` and the
  `sshd` setup used by `b200_config/gen_host.sh`.

### 1.3 Submit a job to the B200 cluster

```bash
cd examples/orchard_swe/b200_config
# Edit the YAML first: replace xxxx with your user name
kubectl create -f 1node_gpu_job_slime-v0.3.0.yaml     # or 2nodes_gpu_job_slime-v0.3.0.yaml
```

## 2. Model conversion

Megatron training reads a `torch_dist` checkpoint, so convert the HF checkpoint once before the first run.

```bash
cd /root/slime
source scripts/models/qwen3.5-35B-A3B.sh

# HF -> torch_dist (initial conversion)
PYTHONPATH=/root/Megatron-LM torchrun \
    --nproc_per_node=8 \
    tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint /data/users/shared/models/Qwen/Qwen3.5-35B-A3B \
    --save /data/users/shared/models/Qwen/Qwen3.5-35B-A3B_torch_dist_slime-0.3.0
```

To **continue training** from a checkpoint that slime already exported to HF format, use the
unfused-experts converter instead:

```bash
source scripts/models/qwen3.5-35B-A3B.sh
PYTHONPATH=/root/Megatron-LM torchrun \
    --nproc_per_node=8 \
    tools/convert_hf_to_torch_dist_unfused_experts.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint /data/users/xxxx/results/saves/orchard-swe-Qwen3.5-35B-A3B/iter_xx_hf \
    --save /data/users/xxxx/models/Qwen/orchard-swe-Qwen3.5-35B-A3B_iter_xx_torch_dist_slime-0.3.0
```

## 3. Prepare data

`scripts/prepare_data.py` downloads a dataset and writes the training and testing JSONL files consumed by
the rollout. For a quick start, example train/eval JSONL files are also provided under
`examples/orchard_swe/swe_data/` (e.g. `mix/swerebench_v2_scaleswe_combined_0514_v4.jsonl` for training and
`swebench/swebench_200_random.jsonl` for evaluation).

```bash
python examples/orchard_swe/scripts/prepare_data.py \
    --output-dir /data/users/xxxx/workspace/swe_data/swebench \
    --dataset princeton-nlp/SWE-bench_Verified

python examples/orchard_swe/scripts/prepare_data.py \
    --output-dir /data/users/xxxx/workspace/swe_data/scaleswe \
    --dataset AweAI-Team/Scale-SWE

python examples/orchard_swe/scripts/prepare_data.py \
    --output-dir /data/users/xxxx/workspace/swe_data/swerebench \
    --dataset nebius/SWE-rebench

python examples/orchard_swe/scripts/prepare_data.py \
    --output-dir /data/users/xxxx/workspace/swe_data/swerebench_v2 \
    --dataset nebius/SWE-rebench-V2
```

## 4. Configure paths

Edit the model / data / load / save paths at the top of the launcher you plan to use in
`examples/orchard_swe/scripts/`. The main variables are:

| Variable | Meaning |
| --- | --- |
| `SAVE_MODEL_ROOT` | Where Megatron checkpoints are written |
| `TEST_DATA_ROOT` | Directory produced by `prepare_data.py` |
| `SWE_TRAJECTORY_DIR` | Where rollout trajectories are dumped |
| `SWE_CONFIG_PATH` | mini-swe-agent config, e.g. `examples/orchard_swe/configs/swe_agent_v2_qwen3.5.yaml` |
| `--hf-checkpoint` / `--ref-load` | HF model dir and its `torch_dist` conversion from step 2 |

## 5. Launch training

Export the required credentials and sandbox endpoints first:

```bash
export WANDB_BASE_URL=https://xxxx.wandb.io
export WANDB_API_KEY=xxxxx
export SANDBOX_API_KEY="abcxyz"
export ORCHARD_SANDBOX_ENDPOINT="http://xx.xx.xx.xx"
```

> Do not hard-code `WANDB_API_KEY` / `SANDBOX_API_KEY` into scripts or commit them; pass them in as
> environment variables or Kubernetes secrets.

Then pick a launcher in `examples/orchard_swe/scripts/`:

| Script | Recipe |
| --- | --- |
| `train_swe_qwen3.5_35B_A3B_multi-node_grpo.sh` | Plain GRPO |
| `train_swe_qwen3.5_35B_A3B_multi-node_grpo_dynamic.sh` | GRPO with progressive/dynamic sampling |
| `train_swe_qwen3.5_35B_A3B_multi-node_grpo_cotograder.sh` | GRPO with rubric-based process reward (RPR) |
| `train_swe_qwen3.5_35B_A3B_multi-node_ppo.sh` | PPO |
| `run-qwen3.5-35B-orchard-swe-sft.sh` | SFT warm-up |
| `run-qwen3.5-35B-orchard-swe-opd.sh` | On-policy distillation (see `README_OPD.md`) |

```bash
bash examples/orchard_swe/scripts/train_swe_qwen3.5_35B_A3B_multi-node_grpo_dynamic.sh
```

All rollout trajectories are written to the directory set in `SWE_TRAJECTORY_DIR`.

### Rubric-based process reward (RPR)

RPR additionally requires GitHub CLI authentication:

```bash
apt-get install gh -y && gh auth login --web -h github.com
bash examples/orchard_swe/scripts/train_swe_qwen3.5_35B_A3B_multi-node_grpo_cotograder.sh
```

### Monitoring and cleanup

```bash
# Monitor Azure CPU usage (run ~20s after training starts)
python examples/orchard_swe/analysis/azure_cpu_monitor.py --ip xx.xx.xx.xx

# If you interrupt training with Ctrl+C, remember to clean up leftover sandboxes.
# This script finds all sandbox ids under /tmp/.orchard_sandboxes and deletes them.
python examples/orchard_swe/scripts/cleanup_azure_sandboxes.py
```

## 6. Convert checkpoints back to HF

```bash
# add --remove-after-conversion if want to remove the original ckpt to save space
python examples/orchard_swe/scripts/convert_all_dist_ckpt_to_hf.py \
  --model-ckpt-parent-dir /data/users/xxxx/ckpt/Qwen3.5-35B-A3B/ \
  --ending-flag 7rlw76 \
  --origin-hf-dir /data/users/shared/models/Qwen/Qwen3.5-35B-A3B
```

## Key hyperparameters

### Rollout throughput and stability

- **`--sglang-max-running-requests`** — maps to SGLang's server argument `--max-running-requests` (slime
  forwards SGLang args by adding the `--sglang-` prefix). It is the maximum number of requests actively
  running inside the SGLang runtime at the same time, i.e. currently prefilling and/or decoding, holding KV
  cache, and participating in scheduling. Each running request consumes GPU memory (KV cache) and
  scheduler/compute budget, so capping it is the primary way to prevent GPU OOM during decoding, keep
  latency from exploding under load, and stabilize throughput by avoiding oversubscription.
- **`--sglang-server-concurrency`** — a slime-specific cap that prevents the SGLang HTTP server from seeing
  too many concurrent requests and crashing. slime can generate very high fan-out (many rollout workers / DP
  ranks / engines); even if the runtime only runs N requests, flooding the HTTP layer with far more in-flight
  requests can cause instability (too many open connections, request bookkeeping overhead, backpressure
  issues). It also helps limit the number of concurrent environment interactions.

### Dynamic sampling (used by `*_grpo_dynamic.sh`)

| Variable | Meaning |
| --- | --- |
| `N_SAMPLES` | Training group size per prompt (`--n-samples-per-prompt`) |
| `N_SAMPLES_MAX` | Maximum samples generated per prompt |
| `N_SAMPLES_STRIDE` | Samples generated per stride before the stop condition is re-checked |
| `POS_RATIO_MIN` / `POS_RATIO_MAX` | Required fraction of positive rewards in the training group |
| `GLOBAL_BATCH_SIZE` | Should equal `ROLLOUT_BATCH_SIZE * N_SAMPLES` |

### SWE agent timeouts and retries

Set via `SWE_*` environment variables in the launcher, e.g. `SWE_TIMEOUT_CREATE_ENV`,
`SWE_TIMEOUT_LLM_INFERENCE`, `SWE_TIMEOUT_GET_OBSERVATION`, `SWE_TIMEOUT_REWARD_EXECUTE`,
`SWE_MAX_ENV_RETRIES` and `SWE_ENV_RETRY_WAIT`, plus the reward shaping knobs `SWE_REWARD_MODE` and
`SWE_UNRESOLVED_REWARD`.

## Analysis tools

Scripts in `examples/orchard_swe/analysis/` help inspect trajectories, rewards and timing:

| Script | What it does |
| --- | --- |
| `cal_val_rewards.py` | Average validation rewards from trajectory outputs; matches reward files against an instance JSONL to compute per-instance and overall metrics |
| `cal_train_rewards.py` | Same idea, for training rollouts |
| `time_analysis.py` | Breaks down LLM inference time vs. environment execution time from trajectory JSON files |
| `azure_cpu_monitor.py` | Live CPU utilization of the sandbox host during training |

Examples:

```bash
# Validation rewards for a trajectory folder against an instance JSONL
python examples/orchard_swe/analysis/cal_val_rewards.py \
  --instance-jsonl /data/users/xxxx/workspace/swe_data/swebench/swebench_200_random.jsonl \
  --reward-folder /data/users/xxxx/workspace/swe_trajectories_qwen3.5/20260729_061226_qwen3.5-35b-a3b_grpo_dynamic_7rlw76/ \
  --rl-step-subfolders

# Time breakdown (LLM inference vs. environment execution)
python examples/orchard_swe/analysis/time_analysis.py \
  --json_folder /data/users/xxxx/workspace/swe_trajectories_qwen3.5/20260729_061226_qwen3.5-35b-a3b_grpo_dynamic_7rlw76/rl_step_0/
```
