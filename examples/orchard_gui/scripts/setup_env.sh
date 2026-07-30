#!/bin/bash
# ==============================================================================
# One-shot setup for the orchard_gui release.
#
#   1. remove the existing slime/ package (if any)
#   2. clone upstream slime at the pinned commit, keep everything except
#      examples/ (ours is already here)
#   3. apply the orchard patch (scripts/slime_f27ef35c.patch) on top
#   4. verify the sandbox orchestrator is present at orchard/orchard_env/
#   5. pip install requirements + Playwright Chromium
#
# Usage (from orchard/trainer/slime/):
#   bash examples/orchard_gui/scripts/set_env.sh
# ==============================================================================
set -euo pipefail

SLIME_REPO="https://github.com/THUDM/slime.git"
SLIME_COMMIT="f27ef35c99e4477fe1f8be8e2f48fcb8d4669f5d"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLIME_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"   # -> orchard/trainer/slime
PATCH_FILE="${SCRIPT_DIR}/slime_f27ef35c.patch"

# 1. Drop the existing slime/ package — replaced by the pinned clone below.
rm -rf "${SLIME_ROOT}/slime"

# 2. Fetch the pinned commit; copy everything except examples/ into the root.
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT
git init -q "${TMP_DIR}"
git -C "${TMP_DIR}" remote add origin "${SLIME_REPO}"
git -C "${TMP_DIR}" fetch -q --depth 1 origin "${SLIME_COMMIT}"
git -C "${TMP_DIR}" checkout -q FETCH_HEAD
rm -rf "${TMP_DIR}/examples" "${TMP_DIR}/.git"
cp -a "${TMP_DIR}/." "${SLIME_ROOT}/"

# 3. Re-apply our modifications on top of the pinned commit.
#    GIT_CEILING_DIRECTORIES: if this tree sits inside a parent git repo,
#    git apply would resolve paths against that repo's toplevel and silently
#    skip every file — pin path resolution to SLIME_ROOT instead.
(cd "${SLIME_ROOT}" && \
    GIT_CEILING_DIRECTORIES="$(dirname "${SLIME_ROOT}")" \
    git apply --whitespace=nowarn "${PATCH_FILE}")

# 4. The sandbox orchestrator ships at orchard/orchard_env/; sandbox_env.py
#    imports it from there directly (no symlink / vendored copy) — just check
#    it is in place. Also drop the env/sandbox symlink older setups created.
ORCHARD_ENV_REPO="${SLIME_ROOT}/../../orchard_env"
[ -e "${ORCHARD_ENV_REPO}/orchard_env/client/sandbox_client.py" ] \
    || echo "WARNING: orchard_env/ not found at orchard/orchard_env — sandbox mode will not work."
SANDBOX_LINK="${SLIME_ROOT}/examples/orchard_gui/env/sandbox"
[ -L "${SANDBOX_LINK}" ] && rm -f "${SANDBOX_LINK}"

# 5. Dependencies.
pip install -r "${SLIME_ROOT}/examples/orchard_gui/requirements.txt"
playwright install chromium

echo "Setup complete: slime @ ${SLIME_COMMIT:0:8} + orchard patch."
