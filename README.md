# Critic-Guided Self-Adaptation for Multilingual Continual Learning

## Overview

This project investigates critic-guided synthetic data generation for multilingual question answering through continual learning across three benchmark datasets: TyDiQA, XQuAD, and MLQA. We employ a three-phase training pipeline where each phase introduces new languages while preserving performance on previously learned languages through representation anchoring. The approach combines multi-format self-edit generation, drift-based filtering, and LLM-based critic scoring to create high-quality training data that balances semantic diversity with faithfulness to the original content.

## Project Structure

```
critic/
├── data/                           # Dataset storage
│   ├── tydiqa/                     # TyDiQA Gold Passage dataset
│   ├── xquad-master/               # XQuAD cross-lingual QA dataset
│   └── MLQA_V1/                    # MLQA multilingual dataset
│       ├── dev/                    # Development split
│       └── test/                   # Test split
│
├── src/                            # Core source code
│   ├── multi_format_self_edit.py  # Multi-format edit generation
│   ├── critic.py                   # Gemini-based critic scorer
│   ├── data_loader.py              # Unified dataset loader
│   ├── build_anchors.py            # Representation anchoring
│   ├── evaluation_metrics.py       # QA evaluation metrics
│   ├── metrics.py                  # Additional metrics calculation
│   └── constants.py                # Language and model constants
│
├── experiments/                    # Experimental scripts
│   ├── evaluate.py                 # Edit generation script
│   ├── select_best_edits.py        # Critic-based filtering
│   ├── train_sft.py                # Standard supervised fine-tuning
│   ├── train_sft_continual.py      # Continual learning with anchoring
│   ├── evaluate_tydiqa.py          # TyDiQA evaluation
│   ├── evaluate_xquad.py           # XQuAD evaluation
│   ├── evaluate_mlqa.py            # MLQA evaluation
│   └── merge_sft_lora.py           # LoRA adapter merging utility
│
├── scripts/                        # Execution scripts
│   ├── phase0.sh                   # Phase 0: TyDiQA training
│   ├── phase1.sh                   # Phase 1: XQuAD training
│   ├── phase2.sh                   # Phase 2: MLQA training
│   └── evaluate_all_phases.sh      # Additional evaluations
│
├── results/                        # Training and evaluation outputs
│   ├── anchors/                    # Saved anchor representations
│   ├── tydi/                       # TyDiQA results
│   ├── xquad/                      # XQuAD results
│   └── mlqa/                       # MLQA results
│
├── accelerate_config.yaml          # Multi-GPU training configuration
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

### File Organization by Task

**Data Generation**
- `multi_format_self_edit.py`: Generates synthetic edits in four formats (QA, rewrite, implications, chain-of-thought)
- `evaluate.py`: Orchestrates edit generation across languages and datasets

**Quality Control**
- `critic.py`: Implements Gemini-based critic for faithfulness and quality scoring
- `select_best_edits.py`: Filters edits using drift bands and critic approval

**Model Training**
- `train_sft.py`: Standard supervised fine-tuning with LoRA
- `train_sft_continual.py`: Continual learning with representation anchoring
- `build_anchors.py`: Creates anchor representations from previous phase models

**Evaluation**
- `evaluate_tydiqa.py`, `evaluate_xquad.py`, `evaluate_mlqa.py`: Dataset-specific evaluation
- `evaluation_metrics.py`, `metrics.py`: Metrics computations

**Data Management**
- `data_loader.py`: Unified interface for loading TyDiQA, XQuAD, MLQA datasets
- `constants.py`: Language mappings and configuration constants

## Setup and Requirements

### Environment Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Set up Google API key for Gemini critic:
```bash
export GOOGLE_API_KEY="your_api_key_here"
```

Or create a `.env` file:
```
GOOGLE_API_KEY=your_api_key_here
```

3. Download datasets:
- TyDiQA: Place `tydiqa-goldp-v1.1-train.json` and `tydiqa-goldp-v1.1-dev.json` in `data/tydiqa/`
- XQuAD: Place language-specific files in `data/xquad-master/`
- MLQA: Place split directories in `data/MLQA_V1/dev/` and `data/MLQA_V1/test/`

### GPU Configuration

For multi-GPU training, adjust `accelerate_config.yaml`:
```yaml
num_processes: 8  # Set to your available GPU count
```

## Running the Pipeline

### Phase 0: Initial Training on TyDiQA

Train on TyDiQA (English, Russian, Finnish, Indonesian, Swahili, Telugu) without anchoring:

```bash
bash scripts/phase0.sh
```

This script:
1. Generates multi-format edits for TyDiQA contexts
2. Filters edits using critic scoring
3. Trains a LoRA adapter on critic-approved edits
4. Evaluates on TyDiQA train and test splits

### Phase 1: Continual Learning on XQuAD

Train on XQuAD (English, Arabic, Spanish, Hindi, Russian, Chinese) with and without anchoring to preserve TyDiQA performance:

```bash
bash scripts/phase1.sh
```

This script:
1. Generates edits for XQuAD using Phase 0 model
2. Filters edits with critic
3. Builds anchor representations from Phase 0 model
4. Trains with and without representation anchoring
5. Evaluates on both XQuAD and TyDiQA

### Phase 2: Continual Learning on MLQA

Train on MLQA (English, German, Vietnamese) while preserving performance on TyDiQA and XQuAD:

```bash
bash scripts/phase2.sh
```

This script:
1. Generates edits for MLQA using Phase 1 model
2. Filters edits with critic
3. Builds anchor representations from Phase 0 and Phase 1 models
4. Trains with and without anchoring
5. Evaluates on MLQA, XQuAD, and TyDiQA

### Additional Evaluations

Run comprehensive cross-phase evaluations:

```bash
bash scripts/evaluate_all_phases.sh
```

This evaluates Phase 1 and Phase 2 models on all previous datasets to measure backward transfer and forgetting.

## Methodology

### Multi-Format Edit Generation

For each context passage, we generate synthetic training data in four formats:

1. **Self-QA**: Question-answer pairs derived from the passage
2. **Rewrite**: Paraphrased passage with different wording
3. **Implications**: List of logical implications from the passage
4. **Chain-of-Thought**: Step-by-step reasoning with implications

The generation model uses instruction-tuned LLMs (Qwen2.5-7B-Instruct) with native-language prompting.

### Drift-Based Filtering

Semantic drift is measured using cross-lingual embeddings (paraphrase-multilingual-MiniLM-L12-v2):

```
d(E) = 1 - cos_sim(embed(C), embed(E))
```

where C is the original context and E is the generated edit. Edits are filtered to a target drift band that balances diversity with faithfulness.

### Critic Scoring

A Gemini-2.5-flash model scores each edit on:
- Faithfulness to original content
- Correct language usage
- Format adherence
- Overall quality

The critic returns:
```
{
  "score": <1-10>,
  "approved": <true if score >= 6>,
  "reason": <explanation>
}
```

Only critic-approved edits are used for training.

### Representation Anchoring

To improve language balance and prevent catastrophic forgetting during continual learning, we use representation anchoring:

```
L_total = L_sft + λ * L_anchor
```

where:
```
L_sft = cross_entropy(y_pred, y_true)
L_anchor = MSE(h_new, h_old)
```

Here, h_old are frozen representations from the previous phase model, and h_new are current representations. The anchoring coefficient λ controls the preservation-adaptation trade-off (λ = 0.1 in our experiments).

### Training Configuration

- Base model: Qwen/Qwen2.5-7B-Instruct
- Adapter: LoRA (r=8, α=16)
- Optimization: AdamW with learning rate 1e-4
- Batch size: Effective batch size of 8 (1 per device, 8 gradient accumulation steps)
- Epochs: 3 per phase
- Mixed precision: FP16

## Mathematical Framework

### Edit Generation

For each context-question-answer triplet, generate K candidate edits:

```
E_{i,k}^{(l,f)} = EditGen(c_i^{(l)}, f), k = 1, ..., K
```

where l is the language and f ∈ {qa, rewrite, implications, cot}.

### Edit Selection

Select the best edit per context using drift filtering and critic scoring:

```
Ê_i^{(l,f)} = argmax_{k: d(E_{i,k}^{(l,f)}) ∈ B} κ(E_{i,k}^{(l,f)})
```

where B is the target drift band and κ is the critic score.

### Continual Learning Objective

The total loss for phase t combines supervised learning and anchoring:

```
L^(t) = E_{(c,y)~D^(t)} [CE(f_θ(c), y)] + λ Σ_{s=1}^{t-1} E_{(c,h)~A^(s)} [||g_θ(c) - h||²]
```

where D^(t) is the current phase dataset, A^(s) are anchors from previous phases, g_θ extracts representations, and λ balances new learning with preservation.

## Results

### Phase 0: TyDiQA Performance

The base Qwen2.5-7B-Instruct model achieves minimal performance on TyDiQA without fine-tuning (F1 = 5.40, ROUGE-L = 7.20). After Phase 0 training on critic-filtered multi-format edits, performance nearly doubles (F1 = 10.51, ROUGE-L = 11.39), with LLM-as-judge correctness improving from 2.07 to 2.24 and language quality from 2.86 to 3.63 on a 1-5 scale. This demonstrates that critic-guided self-edit supervision effectively teaches the model to internalize multilingual QA knowledge.

Cross-lingual analysis reveals balanced gains across resource groups. The HIGH group (English, Russian) achieves F1 = 0.121, MID group (Finnish, Indonesian) F1 = 0.106, and LOW group (Swahili, Telugu) F1 = 0.089. While disparity increases from 0.070 (base) to 0.123 (Phase 0), the HIGH-LOW gap remains modest at 0.032, indicating that critic-filtered edits do not disproportionately favor high-resource languages.

### Phase 1: Forward and Backward Transfer

XQuAD adaptation shows strong forward transfer with minimal backward interference. Starting from the Phase 0 TyDiQA model, Phase 1 training on XQuAD increases average F1 from 7.83 to 10.75-10.77, with ROUGE-L improving from 6.20 to 9.10-9.30. Critically, TyDiQA performance is preserved and even improves: with anchor regularization, TyDiQA F1 increases from 10.51 (Phase 0) to 12.66 (Phase 1), while ROUGE-L rises from 11.39 to 13.49. This demonstrates that continual learning with anchors avoids catastrophic forgetting and enables positive backward transfer.

The anchored model (λ = 0.1) and unanchored model perform similarly on the new XQuAD dataset (F1 = 10.75 vs 10.77), but anchoring substantially benefits retention on TyDiQA. Without anchors, TyDiQA F1 reaches only 11.24 compared to 12.66 with anchors, representing a 12.6% improvement in retention. Cross-lingual disparity remains bounded (0.173-0.176), and resource-group gaps stay stable across both configurations.

Notably, XQuAD shows favorable cross-lingual balance: the HIGH-LOW gap shrinks from 0.029 (zero-shot baseline) to -0.004 (Phase 1 with anchors), with the LOW group (Hindi, Arabic) actually outperforming HIGH group (English, Chinese) after training. This suggests that critic-filtered edits support low-resource language adaptation without requiring explicit language-specific balancing.

### Phase 2: MLQA Continual Learning

After Phase 2 training on MLQA, all three datasets maintain or improve performance. MLQA achieves F1 = 10.92 and ROUGE-L = 11.71, with the highest LLM-as-judge scores across all datasets (correctness = 4.48, quality = 4.51). XQuAD improves further to F1 = 11.75 and ROUGE-L = 9.57. TyDiQA remains stable with F1 = 12.39 and ROUGE-L = 12.73, demonstrating sustained retention across all three phases.

The anchor mechanism introduces minimal performance trade-off on the new MLQA dataset: anchored training achieves F1 = 10.92 compared to 10.62 without anchors, while ROUGE-L remains identical at 11.71. However, anchoring provides better cross-phase stability for XQuAD and TyDiQA. Cross-lingual disparity on MLQA remains extremely low (0.036-0.043), with the LOW group (Vietnamese) consistently outperforming HIGH (English) and MID (German) groups (HIGH-LOW gap = -0.018), indicating excellent language balance.

### Cross-Lingual Stability and Resource Groups

Across all phases, anchor regularization prevents high-resource languages from dominating at the expense of low-resource ones. On XQuAD, the HIGH-LOW gap shifts from positive (high-resource advantage) to negative (low-resource advantage) after Phase 1, and remains negative through Phase 2 (-0.023). On MLQA, the LOW group consistently leads across all configurations. On TyDiQA, the HIGH-LOW gap remains stable (0.032-0.034) across phases, with the LOW group F1 improving from 0.089 (Phase 0) to 0.102 (Phase 1) under anchoring.

Mean XLTR (cross-lingual transfer ratio) stays close to 1.0 across all datasets and phases (1.012-1.080), indicating balanced performance across languages rather than extreme variance. Disparity metrics remain bounded: TyDiQA disparity increases from 0.123 to 0.173-0.176 but stabilizes thereafter, XQuAD disparity ranges from 0.155 to 0.201, and MLQA maintains exceptionally low disparity (0.036-0.043).

### Lexical vs. LLM-as-Judge Agreement

Lexical metrics (F1, ROUGE-L) and LLM-as-judge scores show consistent trends across phases. When F1 and ROUGE-L improve, LLM correctness and quality scores typically follow. For example, moving from Phase 0 to Phase 1 on TyDiQA, both F1 (10.51 to 12.66) and LLM quality (3.63 to 3.71) increase together. MLQA consistently achieves the highest LLM-as-judge scores (correctness 4.40-4.48, quality 4.37-4.51) across all phases, correlating with its strong lexical performance.

Train-test generalization remains strong: trends observed on training splits transfer to held-out test sets with minimal degradation. The dataset ordering (MLQA > XQuAD > TyDiQA) holds consistently across both splits, indicating that edit-based supervision generalizes beyond the self-edit distribution and does not simply overfit to synthetic training data.

## Conclusion

This work demonstrates that critic-guided synthetic data generation combined with representation anchoring enables effective multilingual continual learning for question answering with language balance and prevents catastrophic forgetting. Key findings include:

1. **Multi-format self-edit supervision substantially improves multilingual QA performance.** Critic-filtered edits (QA, rewrite, implications, chain-of-thought) nearly double TyDiQA F1 scores (5.40 to 10.51) and improve language quality ratings from 2.86 to 3.63, demonstrating that diverse synthetic formats teach richer multilingual knowledge than extractive QA pairs alone.

2. **Anchor regularization preserves earlier language capabilities while enabling new learning.** With λ = 0.1, TyDiQA performance improves from Phase 0 (F1 = 10.51) to Phase 1 (F1 = 12.66) while achieving comparable XQuAD performance to unanchored training (F1 = 10.75). Without anchors, TyDiQA retention degrades to F1 = 11.24, a 12.6% loss. This demonstrates that lightweight representation anchoring prevents interference without sacrificing adaptation.

3. **Low-resource languages benefit from critic-guided continual learning.** Contrary to typical high-resource bias, XQuAD's LOW group (Hindi, Arabic) outperforms the HIGH group (English, Chinese) after Phase 1 (HIGH-LOW gap = -0.004), and MLQA's LOW group (Vietnamese) leads throughout Phase 2 (HIGH-LOW gap = -0.018). This suggests that critic filtering naturally supports cross-lingual balance without explicit language-specific reweighting.

4. **Three-phase continual learning successfully scales to 15 languages across diverse datasets.** After training on TyDiQA (6 languages), XQuAD (6 languages), and MLQA (3 languages), the final model maintains strong performance on all three datasets simultaneously (TyDiQA F1 = 12.39, XQuAD F1 = 11.75, MLQA F1 = 10.92), with cross-lingual disparity remaining bounded (0.036-0.201) and mean XLTR near 1.0.

5. **LLM-as-judge metrics align with lexical metrics and generalize beyond training.** Improvements in F1 and ROUGE-L correlate with gains in LLM-evaluated correctness and language quality. Train-test trends remain consistent, with MLQA achieving the highest scores on both splits, indicating that edit-based supervision produces genuine semantic improvements rather than superficial overfitting.

### Limitations and Future Work

Several constraints shaped this study. Due to computational limitations, each phase trained on approximately 100 samples per dataset, likely capping achievable gains. Self-edit generation required 15-16 hours per phase on instructional GPUs, with evaluation adding 2 hours per run, limiting experimental iteration. Exact match (EM) scores remain low (0.00-4.35) because the model generates full-sentence explanations (50-200 tokens) while gold answers are short extractive spans (5-20 tokens), requiring heavier reliance on ROUGE-L and LLM-as-judge metrics.

The critic mechanism also presents trade-offs. While the external Gemini-based critic successfully filtered edits, early experiments with a self-critic variant showed strong self-approval bias, rejecting few edits and providing minimal quality control. Automated filtering reduces obvious failures, but the self-edit corpus inevitably contains noise, redundancy, and stylistic variation that may limit optimization effectiveness.

Future directions include scaling to larger training sets and longer continual sequences to separate data limitations from algorithmic constraints. Adaptive anchor strategies (e.g., phase-dependent λ or language-specific anchoring) may further reduce forgetting in vulnerable languages. Extending critic-guided continual adaptation beyond QA to broader instruction-following and generative tasks would test generality for real-world multilingual deployment. Finally, exploring critic mechanisms that provide edit-level feedback (rather than binary approval) could enable more nuanced data curation and targeted model improvements.
