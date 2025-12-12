#!/usr/bin/env python3

import os
import json
import time
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()


class CriticScorer:
    """
    Score edits using LLM critic (Gemini).
    
    Returns structured JSON with:
    - score: 1-10
    - approved: bool (true if score >= 6)
    - reason: string explanation
    """
    
    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: Optional[str] = None):
        """
        Initialize critic.
        
        Args:
            model_name: Gemini model to use (default: gemini-2.5-flash)
            api_key: Google API key (or from env GOOGLE_API_KEY)
        """
        self.model_name = model_name
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")
        
        genai.configure(api_key=api_key)
        
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
        }
        
        self.model = genai.GenerativeModel(model_name, safety_settings=self.safety_settings)
        self.last_call_time = 0
        self.min_delay = 4.5 
    
    def _build_critic_prompt(
        self, 
        context: str, 
        edit_text: str, 
        format_type: str, 
        language: str
    ) -> str:
        """Build the critic evaluation prompt."""
        
        format_descriptions = {
            "implications": "List of implications derived from the passage",
            "rewrite": "Rewritten passage using different words with same meaning",
            "self_qa": "Question-answer pairs based on the passage content",
            "chain_of_thought": "Step-by-step reasoning with implications"
        }
        
        format_desc = format_descriptions.get(format_type, format_type)
        
        prompt = f"""You are evaluating a synthetic training example for multilingual QA.

**Task**: A model read this passage and generated a {format_desc}.

**Language**: {language}

**Original Passage**:
{context}

**Generated Edit**:
{edit_text}

**Evaluate the edit on these criteria**:

1. **Faithfulness**: Does it accurately reflect the passage content without hallucinating facts?
2. **Language**: Is it written mostly in the correct language/script (same as the passage), with only minor English or loanwords if needed?
3. **Format**: Does it follow the expected format ({format_type})?
4. **Quality**: Is it coherent, useful, and well-structured?

**Output JSON only** (no other text):
{{
  "score": <integer 1-10>,
  "approved": <true if score >= 6, else false>,
  "reason": "<brief explanation>"
}}

**Scoring guide**:
- 1-3: Terrible (hallucinations, wrong language, nonsense)
- 4-5: Poor (some issues with faithfulness/language/format)
- 6-7: Acceptable (minor issues but usable)
- 8-9: Good (high quality, faithful, correct language)
- 10: Excellent (perfect faithfulness, language, format)

**Important**: Set `approved: true` only if score >= 6.
"""
        return prompt
    
    def _build_batch_critic_prompt(self, edit_batch: List[Dict]) -> str:
        """
        Build prompt to score multiple edits in one API call.
        
        Args:
            edit_batch: List of dicts with context, edit_text, format_type, language
        
        Returns:
            Prompt that asks for JSON array of scores
        """
        format_descriptions = {
            "implications": "List of implications derived from the passage",
            "rewrite": "Rewritten passage using different words with same meaning",
            "self_qa": "Question-answer pairs based on the passage content",
            "chain_of_thought": "Step-by-step reasoning with implications"
        }
        
        prompt = """You are evaluating multiple synthetic training examples for multilingual QA.

For each example below, evaluate it on these criteria:
1. **Faithfulness**: Does it accurately reflect the passage content without hallucinating facts?
2. **Language**: Is it written mostly in the correct language/script (same as the passage)?
3. **Format**: Does it follow the expected format?
4. **Quality**: Is it coherent, useful, and well-structured?

**Scoring guide**:
- 1-3: Terrible (hallucinations, wrong language, nonsense)
- 4-5: Poor (some issues with faithfulness/language/format)
- 6-7: Acceptable (minor issues but usable)
- 8-9: Good (high quality, faithful, correct language)
- 10: Excellent (perfect faithfulness, language, format)

---

"""
        
        for i, edit_info in enumerate(edit_batch, 1):
            format_desc = format_descriptions.get(edit_info['format_type'], edit_info['format_type'])
            prompt += f"""**EXAMPLE {i}**
Language: {edit_info['language']}
Format: {format_desc}

Original Passage:
{edit_info['context'][:500]}{"..." if len(edit_info['context']) > 500 else ""}

Generated Edit:
{edit_info['edit_text'][:500]}{"..." if len(edit_info['edit_text']) > 500 else ""}

---

"""
        
        prompt += f"""
**Output a JSON array with {len(edit_batch)} objects** (no other text):
[
  {{"score": <integer 1-10>, "approved": <true if score >= 6, else false>, "reason": "<brief explanation>"}},
  {{"score": <integer 1-10>, "approved": <true if score >= 6, else false>, "reason": "<brief explanation>"}},
  ...
]

**CRITICAL**: 
- Return exactly {len(edit_batch)} score objects in the same order as the examples
- Set "approved": true ONLY if score >= 6
- Output ONLY the JSON array, no markdown, no extra text
"""
        return prompt
    
    def score_edits_batch(
        self,
        edit_batch: List[Dict],
        max_retries: int = 3
    ) -> List[Dict]:
        """
        Score multiple edits in one API call (batching).
        
        Args:
            edit_batch: List of dicts with:
                - context: str
                - edit_text: str
                - format_type: str
                - language: str
            max_retries: Number of retries on failure
        
        Returns:
            List of dicts with critic_score, critic_approved, critic_reason
            (same length as edit_batch)
        """
        if not edit_batch:
            return []
        
        # Rate limiting
        elapsed = time.time() - self.last_call_time
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)
        
        prompt = self._build_batch_critic_prompt(edit_batch)
        
        for attempt in range(max_retries):
            try:
                self.last_call_time = time.time()

                response_schema = {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "edit_number": {"type": "integer"},
                            "score": {"type": "integer", "description": "The critic score between 1 and 10."},
                            "approved": {"type": "boolean"},
                            "reason": {"type": "string"}
                        },
                        "required": ["edit_number", "score", "approved", "reason"]
                    }
                }
                
                # Safety settings already configured at model level
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        response_mime_type="application/json",
                        response_schema=response_schema
                    )
                )
                
                # Check if response was blocked by safety filters
                if not response.candidates or not response.candidates[0].content.parts:
                    finish_reason = response.candidates[0].finish_reason if response.candidates else "UNKNOWN"
                    raise ValueError(f"Response blocked by safety filters (finish_reason={finish_reason}). Safety settings may not be working.")
                
                # Parse JSON response
                response_text = response.text.strip()
                
                # Try to extract JSON if wrapped in markdown
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0].strip()
                
                # Find JSON array
                if "[" in response_text and "]" in response_text:
                    start = response_text.index("[")
                    end = response_text.rindex("]") + 1
                    response_text = response_text[start:end]
                
                # Try to parse JSON with fallback for malformed responses
                try:
                    results = json.loads(response_text)
                except json.JSONDecodeError as e:
                    # Gemini returned malformed JSON - try to extract what we can
                    print(f"  JSON parse error: {str(e)[:100]}")
                    
                    # Save problematic response for debugging
                    error_dir = Path("results/critic_errors")
                    error_dir.mkdir(parents=True, exist_ok=True)
                    error_file = error_dir / f"response_error_batch_{int(time.time())}.txt"
                    with open(error_file, 'w', encoding='utf-8') as f:
                        f.write(f"JSON Parse Error: {str(e)}\n\n")
                        f.write("Raw Response:\n")
                        f.write(response_text)
                    
                    # Try to extract using regex
                    import re
                    partial_results = []
                    # Split by likely object boundaries
                    objects = re.split(r'\},\s*\{', response_text)
                    for obj in objects[:len(edit_batch)]:
                        score_match = re.search(r'"score":\s*(\d+)', obj)
                        approved_match = re.search(r'"approved":\s*(true|false)', obj)
                        reason_match = re.search(r'"reason":\s*"([^"]*)"', obj)
                        
                        if score_match:
                            score = int(score_match.group(1))
                            approved = approved_match.group(1) == "true" if approved_match else (score >= 6)
                            reason = reason_match.group(1) if reason_match else "Parse error"
                            
                            partial_results.append({
                                "score": score,
                                "approved": approved,
                                "reason": reason
                            })
                    
                    if len(partial_results) >= len(edit_batch) // 2:
                        # Got at least half, use partial results
                        results = partial_results
                        print(f"  Extracted {len(partial_results)}/{len(edit_batch)} results from malformed JSON")
                    else:
                        # Too few results, fail this attempt
                        raise ValueError(f"JSON parse failed, only extracted {len(partial_results)}/{len(edit_batch)} results")
                
                # Validate we got the right number of results
                if not isinstance(results, list):
                    raise ValueError(f"Expected list, got {type(results)}")
                
                if len(results) != len(edit_batch):
                    print(f"  Warning: Expected {len(edit_batch)} results, got {len(results)}")
                    # Pad with errors if needed
                    while len(results) < len(edit_batch):
                        results.append({"score": 0, "approved": False, "reason": "Missing from response"})
                
                # Process each result
                processed_results = []
                for result in results:
                    # Validate structure
                    if "score" not in result or "approved" not in result or "reason" not in result:
                        raise ValueError(f"Missing required fields in result: {result}")
                    
                    score = int(result["score"])
                    if not (1 <= score <= 10):
                        raise ValueError(f"Score {score} out of range [1,10]")
                    
                    # Enforce approved logic
                    approved = result["approved"]
                    if approved != (score >= 6):
                        approved = (score >= 6)
                    
                    processed_results.append({
                        "critic_score": score,
                        "critic_approved": approved,
                        "critic_reason": result["reason"]
                    })
                
                return processed_results
                
            except Exception as e:
                if attempt == max_retries - 1:
                    # Final attempt failed - return errors for all edits
                    print(f"Batch scoring failed after {max_retries} attempts: {str(e)}")
                    return [{
                        "critic_score": 0,
                        "critic_approved": False,
                        "critic_reason": "",
                        "critic_error": f"Batch failed: {str(e)}"
                    } for _ in edit_batch]
                
                print(f"Batch retry {attempt + 1}/{max_retries} due to: {str(e)}")
                time.sleep(2)  # Longer wait for batch retries
        
        # Should not reach here
        return [{
            "critic_score": 0,
            "critic_approved": False,
            "critic_reason": "",
            "critic_error": "Unknown error"
        } for _ in edit_batch]
    
    def score_edit(
        self,
        context: str,
        edit_text: str,
        format_type: str,
        language: str,
        max_retries: int = 3
    ) -> Dict:
        """
        Score a single edit using the critic.
        
        Args:
            context: Original passage
            edit_text: Generated edit
            format_type: Format type (implications/rewrite/self_qa/chain_of_thought)
            language: Language code
            max_retries: Number of retries on failure
        
        Returns:
            Dict with:
                - critic_score: int (1-10)
                - critic_approved: bool
                - critic_reason: str
                - critic_error: str (if failed)
        """
        # Rate limiting
        elapsed = time.time() - self.last_call_time
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)
        
        prompt = self._build_critic_prompt(context, edit_text, format_type, language)
        
        for attempt in range(max_retries):
            try:
                self.last_call_time = time.time()
                
                # Safety settings already configured at model level
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=0.0,  # Deterministic scoring
                        max_output_tokens=500
                    )
                )
                
                # Parse JSON response
                response_text = response.text.strip()
                
                # Try to extract JSON if wrapped in markdown
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0].strip()
                
                result = json.loads(response_text)
                
                # Validate response structure
                if "score" not in result or "approved" not in result or "reason" not in result:
                    raise ValueError(f"Missing required fields in response: {result}")
                
                score = int(result["score"])
                if not (1 <= score <= 10):
                    raise ValueError(f"Score {score} out of range [1,10]")
                
                # Enforce approved logic
                approved = result["approved"]
                if approved != (score >= 6):
                    # Critic made a mistake; enforce the rule
                    approved = (score >= 6)
                
                return {
                    "critic_score": score,
                    "critic_approved": approved,
                    "critic_reason": result["reason"]
                }
                
            except Exception as e:
                if attempt == max_retries - 1:
                    # Final attempt failed
                    return {
                        "critic_score": 0,
                        "critic_approved": False,
                        "critic_reason": "",
                        "critic_error": f"Failed after {max_retries} attempts: {str(e)}"
                    }
                time.sleep(1)  # Wait before retry
        
        # Should not reach here
        return {
            "critic_score": 0,
            "critic_approved": False,
            "critic_reason": "",
            "critic_error": "Unknown error"
        }


class QualityScorer:
    """
    Compute continuous quality score Q(e) for edits.
    
    Step D implementation:
    Q(e) = c(e) * sim_factor(e) * lang_factor(e)
    
    Where:
    - c(e) = critic_score / 10
      - critic_score ∈ {0} ∪ [1, 10], so c(e) ∈ {0} ∪ [0.1, 1.0]
      - 0 = hard failure (API error, not scored)
      - 1-10 = actual critic scores
    - sim_factor(e) = (sim - 0.4) / (0.98 - 0.4) ∈ [0, 1]
    - lang_factor(e) = language_match_ratio ∈ [0, 1]
    
    Note: Structure validation is handled at generation time by _validate_edit.
    For training, we only use critic_approved edits (score >= 6).
    """
    
    SIM_MIN = 0.4
    SIM_MAX = 0.98
    
    @staticmethod
    def compute_quality_score(edit: Dict) -> float:
        """
        Compute Q(e) for a single edit.
        
        Args:
            edit: Dict with fields:
                - critic_score: int (0 or 1-10)
                - semantic_similarity: float
                - language_match_ratio: float
        
        Returns:
            Q(e) ∈ [0, 1]
        """
        # 1. Normalized critic score
        critic_score = edit.get("critic_score", 0)
        c_e = critic_score / 10.0  # ∈ {0} ∪ [0.1, 1.0]
        
        # 2. Similarity factor (linear scaling in [0.4, 0.98])
        sim = edit.get("semantic_similarity", 0.0)
        sim_factor = (sim - QualityScorer.SIM_MIN) / (QualityScorer.SIM_MAX - QualityScorer.SIM_MIN)
        sim_factor = max(0.0, min(1.0, sim_factor))  # Clamp to [0, 1]
        
        # 3. Language purity factor
        lang_factor = edit.get("language_match_ratio", 0.0)
        lang_factor = max(0.0, min(1.0, lang_factor))  # Clamp to [0, 1]
        
        # Compute Q(e)
        # Note: Structure validation is already handled by _validate_edit
        q_e = c_e * sim_factor * lang_factor
        
        # Clamp to [0, 1] for safety
        return max(0.0, min(1.0, q_e))


class EditFilter:
    """
    Filter edits using per-(language, format) distribution.
    
    Step E implementation:
    - Group edits by (language, format)
    - If group has < 50 edits: keep all
    - If group has >= 50: keep only Q(e) >= median
    """
    
    MIN_GROUP_SIZE = 50
    
    @staticmethod
    def filter_by_quality_distribution(edits: List[Dict]) -> List[Dict]:
        """
        Filter edits using per-(language, format) median threshold.
        
        Args:
            edits: List of edit dicts with:
                - language: str
                - format_type: str
                - quality_score: float (Q(e))
                - is_valid: bool
                - critic_approved: bool
        
        Returns:
            Filtered list of edits
        """
        # Step 1: Pre-filter - only keep valid + critic-approved
        valid_edits = [
            e for e in edits 
            if e.get("is_valid", False) and e.get("critic_approved", False)
        ]
        
        if not valid_edits:
            return []
        
        # Step 2: Group by (language, format)
        groups = defaultdict(list)
        for edit in valid_edits:
            key = (edit["language"], edit["format_type"])
            groups[key].append(edit)
        
        # Step 3: Filter each group by median
        filtered = []
        
        for (lang, fmt), group_edits in groups.items():
            if len(group_edits) < EditFilter.MIN_GROUP_SIZE:
                # Too small - keep all
                filtered.extend(group_edits)
            else:
                # Compute median Q(e)
                q_scores = [e["quality_score"] for e in group_edits]
                median_q = np.median(q_scores)
                
                # Keep only edits >= median
                for edit in group_edits:
                    if edit["quality_score"] >= median_q:
                        filtered.append(edit)
        
        return filtered


class SamplingWeightCalculator:
    """
    Compute sampling weights from quality scores.
    
    Step F implementation:
    w_raw(e) = Q(e)^α
    
    where α >= 1 controls preference for high-Q edits.
    """
    
    def __init__(self, alpha: float = 2.0):
        """
        Initialize weight calculator.
        
        Args:
            alpha: Exponent for quality weighting (default: 2.0 for quadratic)
        """
        if alpha < 1.0:
            raise ValueError(f"alpha must be >= 1.0, got {alpha}")
        self.alpha = alpha
    
    def compute_weights(self, edits: List[Dict]) -> List[float]:
        """
        Compute sampling weights for edits.
        
        Args:
            edits: List of edit dicts with quality_score field
        
        Returns:
            List of weights (same length as edits)
        """
        weights = []
        for edit in edits:
            q = edit.get("quality_score", 0.0)
            w = q ** self.alpha
            weights.append(w)
        
        return weights
    
    def add_weights_to_edits(self, edits: List[Dict]) -> List[Dict]:
        """
        Add sampling_weight field to each edit.
        
        Args:
            edits: List of edit dicts
        
        Returns:
            Same list with sampling_weight added
        """
        weights = self.compute_weights(edits)
        
        for edit, weight in zip(edits, weights):
            edit["sampling_weight"] = weight
        
        return edits

def score_edits_with_critic(
    edits: List[Dict],
    max_per_article: int = 4,
    batch_size: int = 5,
    critic: Optional[CriticScorer] = None,
    verbose: bool = True
) -> List[Dict]:
    """
    Score valid edits with critic using batching, respecting max_per_article cap.
    
    NOTE: This function expects a FLATTENED list of edits.
    
    Args:
        edits: Flattened list of edit dicts (from generate_edit)
        max_per_article: Max edits to score per article (default: 4 for cost control)
        batch_size: Number of edits to score per API call (default: 5)
        critic: CriticScorer instance (created if None)
        verbose: Print progress
    
    Returns:
        Edits with critic scores added
    """
    if critic is None:
        critic = CriticScorer()
    
    # Group by article_id
    article_groups = defaultdict(list)
    for edit in edits:
        article_id = edit.get("article_id", 0)
        article_groups[article_id].append(edit)
    
    scored_edits = []
    all_valid_edits = []  # Collect all edits to score
    edit_to_article_map = {}  # Track which article each edit belongs to
    
    for article_id, article_edits in article_groups.items():
        # Filter to valid edits only
        valid_edits = [e for e in article_edits if e.get("is_valid", False)]
        
        if len(valid_edits) == 0:
            # No valid edits - keep original with no critic score
            scored_edits.extend(article_edits)
            continue
        
        # Cap to max_per_article (pick highest similarity first)
        if len(valid_edits) > max_per_article:
            valid_edits = sorted(
                valid_edits, 
                key=lambda e: e.get("semantic_similarity", 0.0),
                reverse=True
            )[:max_per_article]
        
        # Add to batch list
        for edit in valid_edits:
            all_valid_edits.append(edit)
            edit_to_article_map[id(edit)] = article_id
        
        # Keep track of all article edits (we'll add scores back later)
        scored_edits.extend(article_edits)
    
    # Batch score all valid edits
    total_scored = 0
    num_batches = (len(all_valid_edits) + batch_size - 1) // batch_size
    
    if verbose:
        print(f"Scoring {len(all_valid_edits)} edits in {num_batches} batches (batch_size={batch_size})")
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(all_valid_edits))
        batch = all_valid_edits[start_idx:end_idx]
        
        if verbose:
            print(f"\nBatch {batch_idx + 1}/{num_batches}: Scoring edits {start_idx}-{end_idx-1}...")
        
        # Prepare batch for API
        batch_info = []
        for edit in batch:
            batch_info.append({
                "context": edit["original_context"],
                "edit_text": edit["generated_text"],
                "format_type": edit["format_type"],
                "language": edit["language"]
            })
        
        # Score batch
        batch_results = critic.score_edits_batch(batch_info)

        # time.sleep(6.5)
        
        # Add scores back to edits
        for edit, result in zip(batch, batch_results):
            edit.update(result)
            total_scored += 1
            
            # DEBUG: Print results
            if verbose:
                if "critic_error" in result:
                    print(f"  Edit {start_idx + batch.index(edit)}: ERROR - {result['critic_error'][:100]}")
                else:
                    print(f"  Edit {start_idx + batch.index(edit)}: score={result['critic_score']}, approved={result['critic_approved']}")
    
    if verbose:
        print(f"\nScored {total_scored} edits with critic")
        
        # Show errors if any
        error_count = sum(1 for e in scored_edits if "critic_error" in e)
        if error_count > 0:
            print(f"\nWARNING: {error_count} edits had critic API errors!")
            for e in scored_edits:
                if "critic_error" in e:
                    print(f"   ERROR: {e['critic_error']}")
                    break  # Just show first error
    
    return scored_edits


def compute_quality_scores(edits: List[Dict]) -> List[Dict]:
    """
    Compute Q(e) for all edits.
    
    Args:
        edits: List of edit dicts with critic scores
    
    Returns:
        Edits with quality_score added
    """
    for edit in edits:
        # Only compute Q(e) for critic-approved edits
        if edit.get("critic_approved", False):
            edit["quality_score"] = QualityScorer.compute_quality_score(edit)
        else:
            edit["quality_score"] = 0.0
    
    return edits


def filter_and_weight_edits(
    edits: List[Dict],
    alpha: float = 2.0,
    verbose: bool = True
) -> List[Dict]:
    """
    Complete pipeline: filter by distribution + add sampling weights.
    
    Args:
        edits: List of edit dicts with quality_score
        alpha: Weighting exponent (default: 2.0)
        verbose: Print statistics
    
    Returns:
        Filtered edits with sampling_weight added
    """
    # Filter by per-(language, format) median
    filtered = EditFilter.filter_by_quality_distribution(edits)
    
    if verbose:
        print(f"Filtered: {len(edits)} -> {len(filtered)} edits")
    
    # Add sampling weights
    weight_calc = SamplingWeightCalculator(alpha=alpha)
    filtered = weight_calc.add_weights_to_edits(filtered)
    
    if verbose:
        if filtered:
            weights = [e["sampling_weight"] for e in filtered]
            print(f"Sampling weights: min={min(weights):.4f}, max={max(weights):.4f}, mean={np.mean(weights):.4f}")
    
    return filtered

def process_edits_pipeline(
    edits: List[Dict],
    max_per_article: int = 4,
    batch_size: int = 5,
    alpha: float = 2.0,
    critic: Optional[CriticScorer] = None,
    verbose: bool = True
) -> List[Dict]:
    """
    Complete critic-based processing pipeline with batching.
    
    Steps:
    C. Score with critic (batched)
    D. Compute Q(e)
    E. Filter by distribution
    F. Add sampling weights
    
    Args:
        edits: Flattened list of raw edits from generator
        max_per_article: Max edits to score per article (default: 4)
        batch_size: Number of edits to score per API call (default: 5)
        alpha: Weighting exponent (default: 2.0)
        critic: CriticScorer instance (optional)
        verbose: Print progress
    
    Returns:
        Final filtered edits with sampling weights
    """
    if verbose:
        print("=" * 60)
        print("CRITIC-BASED EDIT PROCESSING PIPELINE (BATCHED)")
        print("=" * 60)
    
    # Step C: Critic scoring
    if verbose:
        print(f"\n[Step C] Scoring edits with critic (batch_size={batch_size})...")
    edits = score_edits_with_critic(edits, max_per_article, batch_size, critic, verbose)
    
    # Step D: Compute Q(e)
    if verbose:
        print("\n[Step D] Computing quality scores Q(e)...")
    edits = compute_quality_scores(edits)
    
    approved_count = sum(1 for e in edits if e.get("critic_approved", False))
    if verbose:
        print(f"Critic-approved: {approved_count}/{len(edits)} edits")
    
    # Step E + F: Filter and weight
    if verbose:
        print("\n[Step E+F] Filtering by distribution and computing weights...")
    final_edits = filter_and_weight_edits(edits, alpha, verbose)
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"FINAL: {len(final_edits)} training examples ready")
        print("=" * 60)
    
    return final_edits

