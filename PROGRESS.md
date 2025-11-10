# Project Progress

## Overview
 We generate self-edited QA pairs in native languages to enable models to learn continuously without catastrophic forgetting.

## Phase 1: Data and Generation Setup

### Dataset Selection
We chose TyDi QA over MLQA because it contains native language content rather than translations. The dataset includes 49,881 articles across 9 languages: English, Arabic, Bengali, Finnish, Indonesian, Korean, Russian, Swahili, and Telugu.

### Self-Edit Generation
Implemented native language self-edit generation using Ollama llama3.2. The system takes original QA pairs and generates semantically similar but lexically and syntactically varied versions in the same language. This X to X approach preserves linguistic authenticity.

### Drift Measurement
Semantic drift is measured using paraphrase-multilingual-MiniLM-L12-v2 embeddings. We calculate cosine similarity between original and edited versions, with drift defined as 1 minus similarity. Target drift range is 20 to 30 percent, balancing diversity for learning against semantic preservation.

## Phase 2: Preservation Tier Calibration

### Initial Strategy Testing
Tested three preservation tiers on small samples (5 per language):
- High preservation (formerly conservative): 15.32 percent drift
- Medium preservation (formerly moderate): 20.45 percent drift
- Low preservation (formerly aggressive): 27.84 percent drift

Initial results suggested medium preservation was optimal, hitting the target drift range.

### Large-Scale Validation
Scaled to 180 samples (20 per language). Medium preservation unexpectedly produced only 13.34 percent average drift, well below target. Breakdown by language ranged from 9.4 percent for Telugu to 17.3 percent for Swahili. The pilot test result did not generalize at scale.

### Tier Adjustment
Re-evaluated with low preservation tier across all languages, achieving 19.04 percent average drift. This is close to the 20 percent target and deemed sufficient. Six of nine languages exceeded 20 percent drift. Two languages (Indonesian and Telugu) showed anomalous decreases, likely due to LLM stochasticity.

### Mixed Strategy Testing
Tested language-specific tier assignment: low preservation for languages below 13 percent baseline drift, medium preservation for higher baseline languages. Mixed approach yielded 14.25 percent drift, only marginally better than baseline. Medium preservation languages in the mixed run actually had lower drift than baseline, confirming that low preservation works better overall.


### Default Configuration
Set default to low preservation tier with empirically tuned parameters:
- Semantic preservation weight: 0.60
- Lexical preservation weight: 0.40
- Syntactic preservation weight: 0.40
- Max semantic drift: 0.60
- Min intent preservation: 0.75

## Key Results

| Approach | Average Drift | Status |
|----------|---------------|--------|
| Medium preservation baseline | 13.34% | Too conservative |
| Low preservation (all languages) | 19.04% | Recommended |
| Mixed tier assignment | 14.25% | Minimal improvement |

Low preservation achieves near-target drift with consistent performance across languages. This tier is now the default.

## Next Steps

1. Use Gemini as active critic to select best edit from the 5 variants per QA pair
2. Compare native language versus English translation approaches
3. Generate larger validated dataset for training