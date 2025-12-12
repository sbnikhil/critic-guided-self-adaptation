#!/bin/bash


set -e  # Exit on error

# Configuration
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
RESULTS_BASE="results"

echo "=========================================="
echo "PHASE 0: GENERATING EDITS and BEST EDITS"
echo "=========================================="
echo ""
python experiments/evaluate.py \
  --dataset tydi \
  --languages en ru fi id sw te \
  --output-dir "$RESULTS_BASE/tydi/multi_format_all" \
  --samples 100 \
  --split train \
  --model "$BASE_MODEL" \
  --batch-size 1

python experiments/select_best_edits.py \
  --input-folder "$RESULTS_BASE/tydi/multi_format_all" \
  --output-folder "$RESULTS_BASE/tydi/best_edits" \
  --languages en ru fi id sw te

echo "=========================================="
echo "PHASE 0: TYDIQA TRAINING (NO ANCHOR)"
echo "=========================================="
echo ""

echo "[1/5] Training on TyDiQA WITHOUT anchoring..."
python experiments/train_sft_continual.py \
    --input-folder "$RESULTS_BASE/tydi/best_edits" \
    --output-dir "$RESULTS_BASE/tydi/sft_only/checkpoints/final" \
    --base-model "$BASE_MODEL" \
    --anchor-dataset "$RESULTS_BASE/anchors/tydiqa_anchors.pt" \
    --languages en ru fi id sw te \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --gradient-accumulation-steps $GRAD_ACCUM \
    --learning-rate $LR \
    --anchor-lambda $LAMBDA_NO_ANCHOR

echo ""
echo "TyDiQA without anchor complete!"
echo "Checkpoint: $RESULTS_BASE/tydi/sft_only/checkpoints/final"
echo ""


echo "=========================================="
echo "EVALUATION on BASE QWEN MODEL on TYDIQA"
echo "=========================================="

python experiments/evaluate_tydi.py \
  --model_path "Qwen/Qwen2.5-7B-Instruct" \
  --output_dir "tydi/baseQwen_train" \
  --split train \
  --training_data_folder "$RESULTS_BASE/tydi/best_edits/multi_format_all" \
  --languages en ru fi id sw te \
  --num_samples 100 \

echo "=========================================="
echo "EVALUATION on BASE QWEN MODEL on TYDIQA"
echo "=========================================="

python experiments/evaluate_tydi.py \
  --model_path "$RESULTS_BASE/tydi/sft_only/checkpoints/final" \
  --output_dir "tydi/eval_phase0_train" \
  --split train \
  --training_data_folder "$RESULTS_BASE/tydi/best_edits/multi_format_all" \
  --languages en ru fi id sw te \
  --num_samples 100 \

python experiments/evaluate_tydi.py \
  --model_path "$RESULTS_BASE/tydi/sft_only/checkpoints/final" \
  --output_dir "tydi/eval_phase0_test" \
  --split dev \
  --languages en ru fi id sw te \
  --num_samples 100 \  


echo "=========================================="
echo "ALL TRAINING and EvaluationCOMPLETE!"
echo "=========================================="



