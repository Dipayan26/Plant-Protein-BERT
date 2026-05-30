#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# train_gpu_configs.sh — pick-a-GPU launcher for ESM-2 150M continued pretraining.
#
# HOW TO USE:
#   1. Run scripts/vast_onstart.sh first (clone + install + data + tokens).
#   2. Find out which GPU you got:   nvidia-smi --query-gpu=name,memory.total --format=csv
#   3. UNCOMMENT exactly ONE block below that matches your GPU.
#      Leave the others commented out.
#   4. Run:   bash scripts/train_gpu_configs.sh
#
# WHY THE NUMBERS CHANGE PER GPU:
#   Total work is fixed by token_budget (150M = 3.5B tokens). "Fastest" = keep the
#   GPU ~85-90% full so throughput (tokens/sec) is maximal:
#     • bigger microbatch  → higher GPU utilization (the real speed lever)
#     • gradient_checkpointing=false → ~20% faster when VRAM allows
#     • bf16-mixed → every cloud GPU (Ampere+) supports it; stabler than fp16
#   When you change the EFFECTIVE batch you MUST also re-derive max_steps,
#   num_training_steps (so the LR schedule spans the run) and rescale lr.
#
#   tokens/step ≈ effective_batch × 155        (effective_batch = batch_size × accum)
#   max_steps   ≈ token_budget / tokens_per_step   (round UP, used as safety ceiling;
#                                                    TokenBudgetCallback stops first)
#   lr          = base_lr × sqrt(eff_batch / 64)    (sqrt rule — safe for continued PT)
#
# TUNING: start with the values below, watch `nvidia-smi`. If VRAM < ~85%, raise
# batch_size and re-derive steps. If OOM, lower batch_size (or turn gc back on).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root, so the data path resolves

# Common to every block (Ampere+); change to 16-mixed only on pre-Ampere cards.
PREC=bf16-mixed
# Match to the pod's vCPU count (`nproc`) so data loading never stalls the GPU.
NW=$(nproc)


# ═══════════════════════════════════════════════════════════════════════════════
#  150M  (budget 3.5B tokens · base lr 1e-5 @ eff-batch 64)
# ═══════════════════════════════════════════════════════════════════════════════

# ── 150M · 24 GB (RTX 3090 / 4090 / A5000) ─────────────  ACTIVE BY DEFAULT ──────
python scripts/adapt_esm2.py +experiment=adapt_esm2_150m \
    model.gradient_checkpointing=true \
    data.batch_size=64 \
    data.num_workers=${NW} \
    training.trainer.precision=${PREC} \
    training.trainer.accumulate_grad_batches=1 \
    training.trainer.max_steps=400000 \
    training.scheduler.num_training_steps=400000 \
    training.optimizer.lr=1e-5
    # eff-batch 64 → ~9.9K tok/step → ~354K real steps. lr unchanged (same eff batch).

# ── 150M · 48 GB (A6000 / A40 / L40S) ────────────────────────────────────────────
# python scripts/adapt_esm2.py +experiment=adapt_esm2_150m \
#     model.gradient_checkpointing=false \
#     data.batch_size=128 \
#     data.num_workers=${NW} \
#     training.trainer.precision=${PREC} \
#     training.trainer.accumulate_grad_batches=1 \
#     training.trainer.max_steps=200000 \
#     training.scheduler.num_training_steps=200000 \
#     training.optimizer.lr=1.4e-5
#     # eff-batch 128 → ~19.8K tok/step → ~177K real steps. lr = 1e-5×√2.

# ── 150M · 80 GB (A100-80 / H100) ────────────────────────────────────────────────
# python scripts/adapt_esm2.py +experiment=adapt_esm2_150m \
#     model.gradient_checkpointing=false \
#     data.batch_size=256 \
#     data.num_workers=${NW} \
#     training.trainer.precision=${PREC} \
#     training.trainer.accumulate_grad_batches=1 \
#     training.trainer.max_steps=100000 \
#     training.scheduler.num_training_steps=100000 \
#     training.optimizer.lr=2e-5
#     # eff-batch 256 → ~39.7K tok/step → ~88K real steps. lr = 1e-5×2.

# ── 150M · 96 GB (RTX PRO 6000 Blackwell / B200) ─────────────────────────────────
#   ⚠️ BLACKWELL COMPATIBILITY: sm_120 GPUs need CUDA ≥12.8 + torch ≥2.7. The vast
#   image here ships torch 2.5.1 / cu124, which has NO Blackwell kernels — training
#   will error ("no kernel image") or fall back to painfully slow. BEFORE using this
#   block, upgrade inside the box:
#       pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128
#   (or rent on a base image already built with torch ≥2.7 + CUDA 12.8). Verify:
#       python -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.is_available())"
# python scripts/adapt_esm2.py +experiment=adapt_esm2_150m \
#     model.gradient_checkpointing=false \
#     data.batch_size=384 \
#     data.num_workers=${NW} \
#     training.trainer.precision=${PREC} \
#     training.trainer.accumulate_grad_batches=1 \
#     training.trainer.max_steps=65000 \
#     training.scheduler.num_training_steps=60000 \
#     training.optimizer.lr=2.4e-5
#     # eff-batch 384 → ~59.5K tok/step → ~59K real steps. lr = 1e-5×√6 ≈ 2.4e-5.
#     # 96 GB has room for batch=512 (lr 2.8e-5, max_steps 50000, nts 44000) —
#     # push there only if val/mlm_loss stays stable; huge batches can hurt
#     # continued-pretraining convergence. Watch nvidia-smi and back off if OOM.


# ═══════════════════════════════════════════════════════════════════════════════
#  MULTI-GPU (e.g. 2-8× A100) — near-linear speedup. Uncomment + edit ONE block
#  above, then ADD these two override lines to its command:
#     training.trainer.devices=-1 \
#     training.trainer.strategy=ddp
#  DDP multiplies effective batch by num_gpus → scale lr by √(num_gpus) and
#  divide max_steps / num_training_steps by num_gpus.
# ═══════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
#  OPTIONAL code-level speedups (biggest on A100/H100). These are NOT CLI flags —
#  they need a one-line edit in scripts/adapt_esm2.py / the model loader:
#     torch.set_float32_matmul_precision("high")          # TF32, ~1.3-1.5× on Ampere+
#     AutoModel.from_pretrained(..., attn_implementation="sdpa")  # FlashAttention path
#  Ask before enabling — they change the training script, not just config.
# ─────────────────────────────────────────────────────────────────────────────
