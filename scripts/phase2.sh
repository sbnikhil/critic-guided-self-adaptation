#!/bin/bash

set -e  # Exit on error

# Configuration
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
RESULTS_BASE="results"

# Training hyperparameters
EPOCHS=3
BATCH_SIZE=1
GRAD_ACCUM=8
LR=1e-4
LAMBDA_WITH_ANCHOR=0.1
LAMBDA_NO_ANCHOR=0.0

echo "=========================================="
echo "PHASE 2: GENERATING EDITS and BEST EDITS"
echo "=========================================="
echo ""

python experiments/evaluate.py \
  --dataset mlqa \
  --languages en de vi \
  --output-dir "$RESULTS_BASE/mlqa/mlqa_edits" \
  --samples 100 \
  --split train \
  --model "$RESULTS_BASE/xquad/sft_only/checkpoints/final" \
  --batch-size 1

python experiments/select_best_edits.py \
  --input-folder "$RESULTS_BASE/mlqa/mlqa_edits" \
  --output-folder "$RESULTS_BASE/mlqa/mlqa_best_edits" \
  --languages en de vi

echo "=========================================="
echo "PHASE 2: BUILDING ANCHORS FROM PHASE 0+1"
echo "=========================================="
echo ""

python src/build_anchors.py \
  --model-path "$RESULTS_BASE/xquad/sft_only/checkpoints/final" \
  --data-folder "$RESULTS_BASE/tydi/best_edits" "$RESULTS_BASE/xquad/xquad_best_edits" \
  --languages en ru fi id sw te ar es hi zh \
  --samples-per-lang 100 \
  --output-path "$RESULTS_BASE/anchors/tydi_xquad_M2.pt"

echo "=========================================="
echo "PHASE 2: MLQA TRAINING WITH ANCHOR"
echo "=========================================="
echo ""

echo "[1/2] Training on MLQA WITH anchoring..."
python experiments/train_sft_continual.py \
    --input-folder "$RESULTS_BASE/mlqa/mlqa_best_edits" \
    --output-dir "$RESULTS_BASE/mlqa/sft_continual/checkpoints/final" \
    --base-model "$RESULTS_BASE/xquad/sft_only/checkpoints/final" \
    --anchor-dataset "$RESULTS_BASE/anchors/tydi_xquad_M2.pt" \
    --languages en de vi \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --gradient-accumulation-steps $GRAD_ACCUM \
    --learning-rate $LR \
    --anchor-lambda $LAMBDA_WITH_ANCHOR

echo ""
echo "MLQA with anchor complete!"
echo "Checkpoint: $RESULTS_BASE/mlqa/sft_continual/checkpoints/final"
echo ""

echo "=========================================="
echo "PHASE 2: MLQA TRAINING WITHOUT ANCHOR"
echo "=========================================="
echo ""

echo "[2/2] Training on MLQA WITHOUT anchoring..."
python experiments/train_sft_continual.py \
    --input-folder "$RESULTS_BASE/mlqa/mlqa_best_edits" \
    --output-dir "$RESULTS_BASE/mlqa/sft_continual_without_anchor/checkpoints/final" \
    --base-model "$RESULTS_BASE/xquad/sft_only/checkpoints/final" \
    --anchor-dataset "$RESULTS_BASE/anchors/tydi_xquad_M2.pt" \
    --languages en de vi \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --gradient-accumulation-steps $GRAD_ACCUM \
    --learning-rate $LR \
    --anchor-lambda $LAMBDA_NO_ANCHOR

echo ""
echo "MLQA without anchor complete!"
echo "Checkpoint: $RESULTS_BASE/mlqa/sft_continual_without_anchor/checkpoints/final"
echo ""


echo "=========================================="
echo "EVALUATION on PHASE 1 MODEL on MLQA"
echo "=========================================="

python experiments/evaluate_mlqa.py \
  --model_path "$RESULTS_BASE/xquad/sft_only/checkpoints/final" \
  --output_dir "mlqa/eval_phase1_train" \
  --split train \
  --training_data_folder "$RESULTS_BASE/mlqa/mlqa_best_edits" \
  --languages en de vi \
  --num_samples 100

echo "=========================================="
echo "EVALUATION on PHASE 2 WITH ANCHOR on MLQA"
echo "=========================================="

python experiments/evaluate_mlqa.py \
  --model_path "$RESULTS_BASE/mlqa/sft_continual/checkpoints/final" \
  --output_dir "mlqa/eval_phase2_train" \
  --split train \
  --training_data_folder "$RESULTS_BASE/mlqa/mlqa_best_edits" \
  --languages en de vi \
  --num_samples 100

python experiments/evaluate_mlqa.py \
  --model_path "$RESULTS_BASE/mlqa/sft_continual/checkpoints/final" \
  --output_dir "mlqa/eval_phase2_test" \
  --split dev \
  --languages en de vi \
  --num_samples 100

echo "=========================================="
echo "EVALUATION on PHASE 2 WITHOUT ANCHOR on MLQA"
echo "=========================================="

python experiments/evaluate_mlqa.py \
  --model_path "$RESULTS_BASE/mlqa/sft_continual_without_anchor/checkpoints/final" \
  --output_dir "mlqa/eval_phase2_no_anchor_train" \
  --split train \
  --training_data_folder "$RESULTS_BASE/mlqa/mlqa_best_edits" \
  --languages en de vi \
  --num_samples 100


echo "=========================================="
echo "ALL TRAINING and Evaluation COMPLETE!"
echo "=========================================="
