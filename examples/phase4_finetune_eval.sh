#!/usr/bin/env bash
# Phase 4: fine-tune the OWT2-pretrained 216M SpikeGPT on WikiText-103 and
# WikiText-2 separately, then report token-level test perplexity vs the paper
# (Table 2: WT-103 test 39.75, WT-2 test 18.01).
#
# Prereq: Phase 3 pretraining is COMPLETE and its best checkpoint exists.
# WikiText bins were prepared with the SAME GPT-NeoX tokenizer as pretraining
# (vocab 50277) so token-level PPL is comparable to the paper.
#
# Fine-tune uses --reset-schedule: load only the model weights from the
# pretrained checkpoint, reset previous_steps=0 (fresh warmup->cosine over
# --steps) and start with a fresh optimizer. Runs are short, so no resume
# wrapper (and --reset-schedule + a fixed pretrained --checkpoint-in would
# restart from pretrain on any resume anyway).
set -euo pipefail
cd "$(dirname "$0")/.."

PRETRAINED="${PRETRAINED:-runs/spikegpt_216m_owt2_10b.best.pt}"  # final Phase-3 best
COMMON_TRAIN=(
  --device cuda
  --reset-schedule
  --val-holdout-tokens 0          # 0 => use --val-bin (else a train tail is held out)
  --batch 16
  --weight-decay 0.1 --grad-clip 1.0 --dropout 0.0
  --amp bf16 --matmul-precision high
  --compile regional --compile-tail --compile-mode max-autotune-no-cudagraphs --compile-warmup
  --wandb --wandb-project myelin
)

if [[ ! -f "$PRETRAINED" ]]; then
  echo "ERROR: pretrained checkpoint not found: $PRETRAINED" >&2
  echo "Set PRETRAINED=<path to Phase-3 best> and re-run." >&2
  exit 1
fi

# --- WikiText-103 (118.7M train tokens; ~7.2k steps/epoch) ---------------------
uv run python examples/train_tiny_spikegpt.py "${COMMON_TRAIN[@]}" \
  --checkpoint-in "$PRETRAINED" \
  --train-bin data/wikitext103/train.bin \
  --val-bin data/wikitext103/validation.bin \
  --steps 15000 --lr 1e-4 --lr-final 1e-5 --warmup-steps 200 --lr-schedule cosine \
  --log-every 200 --eval-every 500 --val-eval-tokens 250000 \
  --checkpoint-out runs/spikegpt_216m_wt103_ft.pt \
  --best-checkpoint-out runs/spikegpt_216m_wt103_ft.best.pt --checkpoint-every 2000 \
  --wandb-run-name wt103_216m_ft

# --- WikiText-2 (2.4M train tokens; ~147 steps/epoch; overfits fast) ----------
uv run python examples/train_tiny_spikegpt.py "${COMMON_TRAIN[@]}" \
  --checkpoint-in "$PRETRAINED" \
  --train-bin data/wikitext2/train.bin \
  --val-bin data/wikitext2/validation.bin \
  --steps 800 --lr 5e-5 --lr-final 5e-6 --warmup-steps 50 --lr-schedule cosine \
  --log-every 25 --eval-every 50 --val-eval-tokens 250000 \
  --checkpoint-out runs/spikegpt_216m_wt2_ft.pt \
  --best-checkpoint-out runs/spikegpt_216m_wt2_ft.best.pt --checkpoint-every 200 \
  --wandb-run-name wt2_216m_ft

# --- Token-level test perplexity (strided, full-context) vs paper -------------
echo "=== WikiText-103 test PPL (paper: 39.75) ==="
uv run python examples/evaluate_spikegpt_checkpoint.py runs/spikegpt_216m_wt103_ft.best.pt \
  --tokens-bin data/wikitext103/test.bin --device cuda --batch 16

echo "=== WikiText-2 test PPL (paper: 18.01) ==="
uv run python examples/evaluate_spikegpt_checkpoint.py runs/spikegpt_216m_wt2_ft.best.pt \
  --tokens-bin data/wikitext2/test.bin --device cuda --batch 16
