# Browser Environments

Four interchangeable ways to run the browser behind rollouts/evaluation. All
modes drive the same Playwright logic ([`web_env.py`](web_env.py)) and expose
the same `reset / step / exit` interface — they differ only in **where
Chromium runs**.

| Mode | Browser runs… | Best for |
|------|---------------|----------|
| **sandbox** *(default)* | fresh K8s pod per task | production training/eval, strong isolation |
| **local** | in the rollout process | dev, debugging, smoke tests |
| **remote** | a long-lived server you host | fixed pool of self-managed browsers |
| **browser-use** | cloud.browser-use.com | captcha-prone / anti-bot sites |

Pick the mode in [`config.yaml`](config.yaml) (`mode:`) or per run with
`--env-mode local|remote|sandbox|browser-use`.

Credentials live in [`.env`](.env) — `source examples/orchard_gui/env/.env`
before running anything.

## Setup

### sandbox (default)

Needs a reachable sandbox **orchestrator** (the K8s platform vendored in
[`sandbox/`](sandbox/)) and a pod image with the base deps.

```bash
# .env must provide:
#   SANDBOX_ORCHESTRATOR_URL, SANDBOX_API_KEY
# config.yaml -> sandbox: image / cpu / memory / block_network
source examples/orchard_gui/env/.env
python -m examples.orchard_gui.tests.test_sandbox   # smoke: full pod lifecycle
```

The env-server **code is injected from this repo at pod-create time** — the
image only supplies python + fastapi/uvicorn + Playwright/Chromium, so code
changes never require an image rebuild. Rebuild
([`docker_server/Dockerfile.browser`](docker_server/Dockerfile.browser), from the
repo root) only when those *dependencies* change, push under a fresh version
tag, and update `sandbox.image`.

### local

```bash
playwright install --with-deps chromium
python -m examples.orchard_gui.tests.test_env       # smoke
python examples/orchard_gui/run_evaluate.py --env-mode local ...
```

No isolation, one machine — keep `--n-parallel` small (2–8).

### remote

Host one or more `env_server` containers yourself and point the trainer at
them. Each container serves **one** browser on port 8100 — scale by running
more containers.

```bash
# build (from repo root) or use the prebuilt qianhuiwu/browser-env-lite:latest
docker build -f examples/orchard_gui/env/docker_server/Dockerfile.browser -t browser-env .
docker run -d -p 8100:8100 --shm-size=2g browser-env

# config.yaml -> remote.server_url: "http://host-a:8100,http://host-b:8100"
python examples/orchard_gui/run_evaluate.py --env-mode remote ...
```

Set `--n-parallel` ≤ the number of servers.

### browser-use

Managed cloud browser, **billed by the minute** (per-session cost is printed).

```bash
# .env must provide: BROWSER_USE_API_KEY
python examples/orchard_gui/run_evaluate.py --env-mode browser-use ...
# sweep leftover sessions:
python -m examples.orchard_gui.env.clients.browser_use_env --cleanup
```

## Usage

Envs are created automatically by the entry points — you normally never touch
them directly, just pick the mode and run:

```bash
source examples/orchard_gui/env/.env

# Evaluation — sweep checkpoints × benchmarks (starts its own SGLang server
# in the `sglang` tmux session; edit hf_checkpoints/task_fns in the script):
bash examples/orchard_gui/scripts/run_evaluate.sh

# ...or a single eval directly (needs a serving SGLang endpoint):
python examples/orchard_gui/run_evaluate.py \
    --hf-checkpoint <ckpt_dir> \
    --task-file examples/orchard_gui/data/online-mind2web.jsonl \
    --eval-protocol online_mind2web --n-parallel 32

# RL training — rollouts create one env per trajectory via the same factory:
bash examples/orchard_gui/scripts/run_browser_qwen3.5_9b.sh
```

Programmatic use: `factory.create_env(config)` returns the mode's env; then
`await env.reset()` → `(obs, info)`, `await env.step(actions)`, `await env.exit()`.
`obs["screenshot"]` is PNG bytes.

## Notes

- `--n-parallel N` = at most N live environments at once (one per in-flight
  task). Sandbox mode is the one built to scale N high.
- Interrupted runs leave no leaks: sandbox pods and browser-use sessions are
  tracked in `.sandboxes/` / `.browser_use_sessions/` and swept on the next run.
- Shared `config.yaml` keys: viewport (`width`/`height`/`dpr`), coordinate
  transform (`resize_scale`, `image_patch_size` — 32 for Qwen3-VL, 28 for
  Qwen2.5-VL), prompts (`path_to_policy`, `path_to_tool_list`), observation
  channels (`use_screenshot`, `use_a11ytree`).
