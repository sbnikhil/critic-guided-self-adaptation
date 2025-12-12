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
echo "PHASE 1: GENERATING EDITS and BEST EDITS"
echo "=========================================="
echo ""

python experiments/evaluate.py \
  --dataset xquad \
  --languages en ar es hi ru zh \
  --output-dir "$RESULTS_BASE/xquad/xquad_edits" \
  --samples 100 \
  --split train \
  --model "$RESULTS_BASE/tydi/sft_only/checkpoints/final" \
  --batch-size 1

python experiments/select_best_edits.py \
  --input-folder "$RESULTS_BASE/xquad/xquad_edits" \
  --output-folder "$RESULTS_BASE/xquad/xquad_best_edits" \
  --languages en ar es hi ru zh

echo "=========================================="
echo "PHASE 1: BUILDING ANCHORS FROM PHASE 0"
echo "=========================================="
echo ""

python src/build_anchors.py \
  --model-path "$RESULTS_BASE/tydi/sft_only/checkpoints/final" \
  --data-folder "$RESULTS_BASE/tydi/best_edits" \
  --languages en ru fi id sw te \
  --samples-per-lang 100 \
  --output-path "$RESULTS_BASE/anchors/tydi_M1.pt"

echo "=========================================="
echo "PHASE 1: XQUAD TRAINING WITH ANCHOR"
echo "=========================================="
echo ""

echo "[1/2] Training on XQuAD WITH anchoring..."
python experiments/train_sft_continual.py \
    --input-folder "$RESULTS_BASE/xquad/xquad_best_edits" \
    --output-dir "$RESULTS_BASE/xquad/sft_only/checkpoints/final" \
    --base-model "$BASE_MODEL" \
    --anchor-dataset "$RESULTS_BASE/anchors/tydi_M1.pt" \
    --languages en ar es hi ru zh \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --gradient-accumulation-steps $GRAD_ACCUM \
    --learning-rate $LR \
    --anchor-lambda $LAMBDA_WITH_ANCHOR

echo ""
echo "XQuAD with anchor complete!"
echo "Checkpoint: $RESULTS_BASE/xquad/sft_only/checkpoints/final"
echo ""

echo "=========================================="
echo "PHASE 1: XQUAD TRAINING WITHOUT ANCHOR"
echo "=========================================="
echo ""

echo "[2/2] Training on XQuAD WITHOUT anchoring..."
python experiments/train_sft_continual.py \
    --input-folder "$RESULTS_BASE/xquad/xquad_best_edits" \
    --output-dir "$RESULTS_BASE/xquad/sft_only_no_anchor_baseline/final/final" \
    --base-model "$BASE_MODEL" \
    --anchor-dataset "$RESULTS_BASE/anchors/tydi_M1.pt" \
    --languages en ar es hi ru zh \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --gradient-accumulation-steps $GRAD_ACCUM \
    --learning-rate $LR \
    --anchor-lambda $LAMBDA_NO_ANCHOR

echo ""
echo "XQuAD without anchor complete!"
echo "Checkpoint: $RESULTS_BASE/xquad/sft_only_no_anchor_baseline/final/final"
echo ""


echo "=========================================="
echo "EVALUATION on PHASE 0 MODEL on XQUAD"
echo "=========================================="

python experiments/evaluate_xquad.py \
  --model_path "$RESULTS_BASE/tydi/sft_only/checkpoints/final" \
  --output_dir "xquad/eval_phase0_train" \
  --split train \
  --training_data_folder "$RESULTS_BASE/xquad/xquad_best_edits" \
  --languages en ar es hi ru zh \
  --num_samples 100

echo "=========================================="
echo "EVALUATION on PHASE 1 WITH ANCHOR on XQUAD"
echo "=========================================="

python experiments/evaluate_xquad.py \
  --model_path "$RESULTS_BASE/xquad/sft_only/checkpoints/final" \
  --output_dir "xquad/eval_phase1_train" \
  --split train \
  --training_data_folder "$RESULTS_BASE/xquad/xquad_best_edits" \
  --languages en ar es hi ru zh \
  --num_samples 100

python experiments/evaluate_xquad.py \
  --model_path "$RESULTS_BASE/xquad/sft_only/checkpoints/final" \
  --output_dir "xquad/eval_phase1_test" \
  --split test \
  --languages en ar es hi ru zh \
  --num_samples 100

echo "=========================================="
echo "EVALUATION on PHASE 1 WITHOUT ANCHOR on XQUAD"
echo "=========================================="

python experiments/evaluate_xquad.py \
  --model_path "$RESULTS_BASE/xquad/sft_only_no_anchor_baseline/final/final" \
  --output_dir "xquad/eval_phase1_no_anchor_train" \
  --split train \
  --training_data_folder "$RESULTS_BASE/xquad/xquad_best_edits" \
  --languages en ar es hi ru zh \
  --num_samples 100


echo "=========================================="
echo "ALL TRAINING and Evaluation COMPLETE!"
echo "=========================================="
