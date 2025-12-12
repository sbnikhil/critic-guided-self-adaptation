#!/bin/bash

set -e  # Exit on error

# Configuration
RESULTS_BASE="results"

echo "=========================================="
echo "ADDITIONAL EVALUATIONS ACROSS ALL PHASES"
echo "=========================================="
echo ""

echo "=========================================="
echo "EVALUATION on PHASE 1 WITH ANCHOR on TYDIQA"
echo "=========================================="
echo ""

echo "[1/3] Phase 1 with anchor - train split..."
python experiments/evaluate_tydiqa.py \
  --model_path "$RESULTS_BASE/xquad/sft_only/checkpoints/final" \
  --output_dir "tydi/eval_phase1_train" \
  --split train \
  --training_data_folder "$RESULTS_BASE/tydi/best_edits" \
  --languages en ru fi id sw te \
  --num_samples 100


echo ""
echo "=========================================="
echo "EVALUATION on PHASE 1 WITHOUT ANCHOR on TYDIQA"
echo "=========================================="
echo ""

echo "[2/3] Phase 1 without anchor - train split..."
python experiments/evaluate_tydiqa.py \
  --model_path "$RESULTS_BASE/xquad/sft_only_no_anchor_baseline/final/final" \
  --output_dir "tydi/eval_phase1_no_anchor_train" \
  --split train \
  --training_data_folder "$RESULTS_BASE/tydi/best_edits" \
  --languages en ru fi id sw te \
  --num_samples 100


echo ""
echo "Phase 1 TyDiQA evaluations complete!"
echo ""

echo "=========================================="
echo "EVALUATION on PHASE 2 WITH ANCHOR on TYDIQA"
echo "=========================================="
echo ""

echo "[2/3] Phase 2 with anchor - train split..."
python experiments/evaluate_tydiqa.py \
  --model_path "$RESULTS_BASE/mlqa/sft_continual/checkpoints/final" \
  --output_dir "tydi/eval_phase2_train" \
  --split train \
  --training_data_folder "$RESULTS_BASE/tydi/best_edits" \
  --languages en ru fi id sw te \
  --num_samples 100


echo ""
echo "Phase 2 TyDiQA evaluations complete!"
echo ""

# =============================================================================
# PHASE 2: XQUAD EVALUATIONS (Phase 1 → Phase 2)
# =============================================================================

echo "=========================================="
echo "EVALUATION on PHASE 2 WITH ANCHOR on XQUAD"
echo "=========================================="
echo ""

echo "[1/1] Phase 2 with anchor - train split..."
python experiments/evaluate_xquad.py \
  --model_path "$RESULTS_BASE/mlqa/sft_continual/checkpoints/final" \
  --output_dir "xquad/eval_phase2_train" \
  --split train \
  --training_data_folder "$RESULTS_BASE/xquad/xquad_best_edits" \
  --languages en ar es hi ru zh \
  --num_samples 100

echo ""
echo "Phase 2 XQuAD evaluations complete!"
echo ""


