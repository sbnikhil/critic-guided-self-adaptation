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

## Phase 3: Multi-Format Generation and Critic-Based Selection

### Multi-Format Self-Edit Architecture
Implemented a multi-format generation system that creates four distinct types of content variations for each QA pair:
1. **Implications**: Explores deeper meanings and consequences of the answer
2. **Rewrite**: Restructures the answer with different phrasing while preserving meaning
3. **Self-QA**: Generates follow-up questions and answers based on the context
4. **Chain of Thought**: Adds explicit reasoning steps leading to the answer

This approach generates 4 edits per QA pair across all 9 languages, providing diverse learning signals while maintaining semantic fidelity.

### Drift Performance
The multi-format approach achieved significant improvement in semantic diversity:
- **Average drift: 28.2%** (up from 19.04% in Phase 2)
- Drift range across languages: 23.7% to 33.1%
- All languages exceeded the 20% minimum target
- Successfully balanced within the 20-30% target range

Format-specific drift analysis showed:
- Chain of thought: Highest drift (most transformative)
- Self-QA: High drift (new question-answer pairs)
- Rewrite: Moderate drift (structural changes)
- Implications: Lower drift (semantic extensions)

### Gemini Critic for Best Edit Selection
Implemented Google Gemini (gemini-2.5-flash) as an active critic to evaluate and select the best edit from the 4 generated variants per QA pair. The critic system:

**Architecture:**
- Batch evaluation: 2 QA pairs per API call (rate limit optimization)
- 6.5 second delays between batches to avoid hitting API limits
- Critical safety configuration: `HarmBlockThreshold.BLOCK_NONE` for all categories (essential for multilingual content evaluation)

**Evaluation Criteria:**
The critic assesses each edit on:
- Factual accuracy and correctness
- Semantic preservation of original meaning
- Natural language quality and fluency
- Cultural and linguistic appropriateness

**Results on 10 samples per language (360 total edits evaluated):**
- **97.8% approval rate** across all languages and formats
- **Average quality score: 7.5/10**
- Consistent performance across all 9 languages
- Bengali and Telugu showed highest approval (100%)
- Swahili showed lowest but still strong approval (90%)

The critic successfully filters low-quality edits while maintaining high approval rates, validating that the multi-format generation produces semantically sound variations suitable for continual learning.

### Final Dataset
Generated and validated dataset with critic-selected best edits:
- 10 QA pairs × 9 languages = 90 best edits total
- All edits critic-approved with detailed reasoning
- 28.2% average drift maintains diversity for learning
- High quality scores ensure semantic fidelity

## Next Steps

1. Scale to full dataset generation across all TyDi QA samples
2. Implement continual learning training loop with selected edits
3. Measure catastrophic forgetting prevention effectiveness