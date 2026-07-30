# 🌐 Orchard GUI — Browser Agent RL with slime

Train and evaluate a web-automation agent (Qwen3-VL / Qwen3.5) with RL in
slime. The agent looks at browser **screenshots**, emits tool calls (click,
type, scroll, …), and is rewarded by an LLM judge for completing real web
tasks.

## 🛠️ Setup

> ⚠️ **Warning:** orchard_gui pins a newer slime version — during setup (step 3),
> `setup_env.sh` **overwrites the existing `slime/` code base** with it. If you
> use this checkout for other work, run orchard_gui from a separate copy of the
> repo root. All commands below run from `orchard/trainer/slime/`.

**1. Prepare your `.env` 🔑**

```bash
cp examples/orchard_gui/env/.env.example examples/orchard_gui/env/.env
# fill in your keys (sandbox orchestrator, LLM judge, wandb, HF)
```

**2. Check the two configs ⚙️**

- [`env/config.yaml`](env/config.yaml) — environment: `mode` (`sandbox` by
  default), sandbox image, viewport, prompts. See [`env/README.md`](env/README.md).
- [`browser_training_config.yaml`](browser_training_config.yaml) — training /
  eval: max steps, judge model and `judge_api_mode`, timeouts.

**3. Run the setup script 📦**

```bash
bash examples/orchard_gui/scripts/setup_env.sh
```

It rebuilds `slime/` from the pinned upstream commit, applies the orchard
patch ([`scripts/slime_f27ef35c.patch`](scripts/slime_f27ef35c.patch)), checks
that the sandbox orchestrator is present at `orchard/orchard_env/` (imported
from there directly by `env/clients/sandbox_env.py`), and installs python
deps + Playwright Chromium.

## 🚀 Run

```bash
source examples/orchard_gui/env/.env

# RL training (model / data / save paths are set at the top of the script)
bash examples/orchard_gui/scripts/run_browser_qwen3vl_4b_thinking.sh
```

### 🔁 Convert a Megatron checkpoint to HF

Training writes Megatron (torch_dist) checkpoints to `${SAVE_DIR}/iter_XXXXXXX`
every `--save-interval` rollouts, and also exports ready-to-serve HF checkpoints
to `${SAVE_DIR}/hf/rollout_{id}` (`--save-hf`). To convert a Megatron iteration
that has no HF export (e.g. to evaluate it):

```bash
PYTHONPATH=/root/Megatron-LM:$PWD python3 tools/convert_torch_dist_to_hf_bridge.py \
    --input-dir  examples/orchard_gui/outputs/Qwen3-VL-4B-Thinking_orchard_gui/iter_0000000 \
    --output-dir examples/orchard_gui/outputs/Qwen3-VL-4B-Thinking_orchard_gui/hf/iter_0000000 \
    --origin-hf-dir examples/orchard_gui/models/Qwen3-VL-4B-Thinking
```

Add `-f` to overwrite an existing output dir.

### 📊 Evaluation

Three benchmarks are supported. Each pairs a task file in [`data/`](data/)
with its own LLM-judge reward function in [`eval/`](eval/), selected via
`--eval-protocol` in [`run_evaluate.py`](run_evaluate.py):

| Benchmark | Task file | Reward function |
|---|---|---|
| Online-Mind2Web | [`data/online-mind2web.jsonl`](data/online-mind2web.jsonl) | [`eval/reward_online_mind2web.py`](eval/reward_online_mind2web.py) — FARA AgentTrek single-call judge |
| WebVoyager | [`data/webvoyager_fara.jsonl`](data/webvoyager_fara.jsonl) | [`eval/reward_webvoyager.py`](eval/reward_webvoyager.py) — FARA-style pure judge |
| DeepShop | [`data/deepshop.jsonl`](data/deepshop.jsonl) | [`eval/reward_deepshop.py`](eval/reward_deepshop.py) — Molmo Web structured-output judge |

First set the HF checkpoint(s) in the `hf_checkpoints` array at the top of
[`scripts/run_evaluate.sh`](scripts/run_evaluate.sh). The script then sweeps
checkpoints × benchmarks, serving each checkpoint with SGLang in a tmux
session named `sglang`:

```bash
tmux new-session -d -s sglang
bash examples/orchard_gui/scripts/run_evaluate.sh
```