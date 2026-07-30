# Minimal OPD + SWE-Reward Run (Qwen3.5-35B-A3B)

This guide covers [`run-qwen3.5-35B-opd-swe-reward-colocate-8gpu-minimal.sh`](run-qwen3.5-35B-opd-swe-reward-colocate-8gpu-minimal.sh),
a stripped-down single-node, 8-GPU **colocated** run that combines:

- **On-policy distillation (OPD)** — the student matches a larger teacher's token-level log-probs, and
- **SWE-bench task reward** — GRPO advantages from actually resolving SWE issues.

Training and rollout share the same 8 GPUs. The teacher runs on a **separate SGLang
server** that you launch yourself; the script only talks to it over HTTP.

All experiment-sweep machinery from the original script has been removed, and every
CLI flag that is no longer supported on this branch has been dropped. The handful of
knobs worth playing with are exposed as environment variables (see below).

## Prerequisites

1. **Student checkpoints** (HF + torch_dist), teacher HF tokenizer, and SWE data at the
   paths hardcoded near the top of the script. Edit these to match your environment:
   - `STUDENT_HF`, `STUDENT_TORCH_DIST`
   - `TEACHER_HF` (only the tokenizer is needed locally)
   - `PROMPT_DATA`, `EVAL_DATA`
2. **A running teacher SGLang server.** Point the script at it:
   - `TEACHER_IP`, `TEACHER_PORT`
3. **W&B key** (optional). Export before running:
   ```bash
   export WANDB_API_KEY=...
   ```

## Running

```bash
bash examples/on_policy_distillation/run-qwen3.5-35B-opd-swe-reward-colocate-8gpu-minimal.sh
```

The script cleans up any prior Ray/SGLang processes, starts a single-node Ray head,
writes a runtime OPD config, and submits the training job.

## Configurable knobs (environment variables)

Override any of these on the command line, e.g.:

```bash
OPD_ONLY=1 OPD_LOSS_TYPE=k3 bash examples/on_policy_distillation/run-qwen3.5-35B-opd-swe-reward-colocate-8gpu-minimal.sh
```

| Env var | Default | Description |
|---------|---------|-------------|
| `OPD_ONLY` | `0` | `1` = **OPD only** (`--zero-train-adv-for-opd`: zero the GRPO task-reward advantage so only the distillation KL trains the student). `0` = **OPD + GRPO** (keep both the SWE reward advantage and the OPD KL). |
| `OPD_KL_COEF` | `1.0` | OPD KL penalty coefficient (`--opd-kl-coef`). Scales the strength of the distillation signal. |
| `OPD_LOSS_TYPE` | `js_skew` | OPD loss surrogate (`--opd-reverse-kl-loss-type`). One of `k1`, `k2`, `k3`, `js_skew`. |
| `OPD_JS_MIXTURE_WEIGHT` | `0.5` | Teacher mixture weight λ ∈ (0, 1] for the `js_skew` loss (`--opd-js-mixture-weight`). Only used when `OPD_LOSS_TYPE=js_skew`. |

### OPD only vs OPD + GRPO

- **OPD only** (`OPD_ONLY=1`): the student is trained purely to imitate the teacher.
  The SWE task reward is still computed (and logged) but its advantage is zeroed, so it
  does not drive gradients. Good baseline for "pure distillation".
- **OPD + GRPO** (`OPD_ONLY=0`, default): the student learns from *both* the teacher's
  log-probs *and* the SWE outcome reward. The two signals are combined — OPD as an added
  loss term (`--opd-mode loss`) and the SWE reward through the GRPO advantage.

### Different OPD losses (`OPD_LOSS_TYPE`)

The loss surrogate controls how the per-token reverse-KL to the teacher is turned into a
differentiable objective:

| Value | What it does |
|-------|--------------|
| `k1` | Score-function estimator: detached reverse-KL coefficient × student log-prob. |
| `k2` | `0.5 · (log π_s − log π_t)²` — squared log-ratio penalty. |
| `k3` | `exp(log π_t − log π_s) − 1 + (log π_s − log π_t)` — the k3 KL estimator. |
| `js_skew` | Skew (generalized) Jensen–Shannon divergence with mixture `m = (1−λ)·π_s + λ·π_t`, where λ = `OPD_JS_MIXTURE_WEIGHT`. Reduces to plain JS at λ=0.5 and interpolates toward reverse-KL as λ→1. |

Example sweeps:

```bash
# Pure distillation, k3 loss
OPD_ONLY=1 OPD_LOSS_TYPE=k3 bash .../run-...-minimal.sh

# OPD + GRPO, skew-JS with a stronger teacher pull
OPD_LOSS_TYPE=js_skew OPD_JS_MIXTURE_WEIGHT=0.8 bash .../run-...-minimal.sh

# Weaker distillation signal alongside GRPO
OPD_KL_COEF=0.3 bash .../run-...-minimal.sh
```

## What is fixed in this script

These are hardcoded (edit the script directly if you need to change them):

- `--opd-mode loss` — OPD is added as a separate loss term (not folded into advantages).
- `--advantage-estimator grpo` with default (std) reward normalization.
- Parallelism: TP=2, CP=4, EP=8, PP=1; dynamic batch size, `--max-tokens-per-gpu 16384`.
- Optimizer: Adam, `lr=1e-6`, constant schedule, precision-aware optimizer.
- Rollout/train batch sizes: `ROLL_BS=8`, `n-samples-per-prompt=8`, `global-batch-size=32`.

## Outputs

- Checkpoints and the copied `run_config.sh` are written under `SAVE_DIR`.
- The best checkpoint (by `SLIME_BEST_CKPT_METRIC=eval/swe_val/resolved_count`) is saved
  under `${SAVE_DIR}/best`.
- Metrics are logged to W&B project `slime-opd-debug`, group `${MODEL_NAME}-opd-swe-minimal`.

## See also

- [`README.md`](README.md) — general OPD example (math data, sglang vs megatron teacher).
- `slime/rollout/opd_config.py` — teacher endpoint / truncation config schema.
- `slime/backends/megatron_utils/loss.py` — the OPD loss surrogates listed above.
