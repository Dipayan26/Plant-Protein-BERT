#!/usr/bin/env bash
# vast.ai on-start / setup script for Plant-Protein-BERT.
# Paste the body into the template's "On-start Script" box, OR run it manually
# after SSHing in.  Idempotent — safe to re-run.
#
# Tokens: set these BEFORE running (e.g. in the vast template's env vars, or
# `export` them in your shell).  Both are optional but recommended:
#   HF_TOKEN        — needed to download the private dataset repo
#   WANDB_API_KEY   — needed for experiment logging (else W&B runs offline)
#
# You can also pass them inline:
#   HF_TOKEN=hf_xxx WANDB_API_KEY=xxx bash scripts/vast_onstart.sh
set -euo pipefail

# # ──────────────────────────────────────────────────────────────────────────────
# # TOKENS
# # ──────────────────────────────────────────────────────────────────────────────
# # SAFEST (recommended): keep secrets OUT of this tracked file. Put them in an
# # untracked file and this script will load them automatically:
# #     printf 'export HF_TOKEN=hf_xxx\nexport WANDB_API_KEY=xxx\n' > /workspace/.secrets
# [ -f /workspace/.secrets ] && source /workspace/.secrets
# #
# # QUICK (NOT for committing): you may paste keys on the two lines below, but then
# # DO NOT `git add`/commit this file, or your tokens will leak publicly on GitHub.
# # Leave them blank to rely on .secrets / exported env vars / vast template vars.
# HF_TOKEN="${HF_TOKEN:-}"
# WANDB_API_KEY="${WANDB_API_KEY:-}"


# Your full sequence on the cloud box:
# export HF_TOKEN=hf_xxx
# export WANDB_API_KEY=xxx
# bash scripts/vast_onstart.sh
# # ──────────────────────────────────────────────────────────────────────────────

REPO_DIR=/workspace/Plant-Protein-BERT
# The config (configs/esm2_adapt/data/trembl_plant.yaml) expects the data at
# <launch-dir>/outputs/processed/trembl_full, so download straight into it —
# then training needs NO data.processed_dir override on any model size.
DATA_DIR="${REPO_DIR}/outputs/processed/trembl_full"

cd /workspace

# 1. Get the code (skip if already cloned).
if [ ! -d "${REPO_DIR}" ]; then
    git clone https://github.com/Dipayan26/Plant-Protein-BERT.git
fi
cd "${REPO_DIR}"

# 2. Register the package so `import plant_bert` works.
pip install -e .

# 3. Authenticate to Hugging Face (for the private dataset).
if [ -n "${HF_TOKEN:-}" ]; then
    echo "Logging in to Hugging Face with HF_TOKEN..."
    hf auth login --token "${HF_TOKEN}" --add-to-git-credential || \
        huggingface-cli login --token "${HF_TOKEN}"
else
    echo "WARNING: HF_TOKEN not set — dataset download may fail if the repo is private."
    echo "         Set it with:  export HF_TOKEN=hf_xxx   (then re-run)"
fi

# 4. Authenticate to Weights & Biases (for logging).
if [ -n "${WANDB_API_KEY:-}" ]; then
    echo "Logging in to Weights & Biases..."
    wandb login "${WANDB_API_KEY}"
else
    echo "NOTE: WANDB_API_KEY not set — W&B will run in offline mode."
    echo "      Set it with:  export WANDB_API_KEY=xxx   (then re-run), or"
    echo "      force offline: export WANDB_MODE=offline"
fi

# 5. Pull the training data straight into the path the config expects.
#    Xet-backed fast transfer (HF_XET_HIGH_PERFORMANCE is set in the image).
mkdir -p "${DATA_DIR}"
echo "Downloading dataset into ${DATA_DIR} ..."
hf download dipayan26/plant-trembl-h5 \
    --repo-type dataset \
    --local-dir "${DATA_DIR}"

# 6. Sanity check: the data module needs sequences.h5 at the top of DATA_DIR.
echo "Contents of ${DATA_DIR}:"
ls -lh "${DATA_DIR}"
if [ ! -f "${DATA_DIR}/sequences.h5" ]; then
    echo "WARNING: sequences.h5 not found at the top of ${DATA_DIR}."
    echo "         If the HF repo nested it in a subfolder, move it up or adjust the path."
fi

echo
echo "Setup complete. Train with (no data path override needed):"
echo "  cd ${REPO_DIR}"
echo "  python scripts/adapt_esm2.py +experiment=adapt_esm2_8m     # or _35m / _150m"
