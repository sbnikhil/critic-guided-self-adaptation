import re
import numpy as np
from typing import List, Dict, Any, Tuple
import time
import asyncio
import os

try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None
try:
    from bleurt import score as bleurt_score
except ImportError:
    bleurt_score = None
try:
    import google.generativeai as genai
except ImportError:
    genai = None


TYDIQA_RESOURCE_GROUPS = {
    'HIGH': ['en', 'ru'],
    'MID': ['fi', 'id'],
    'LOW': ['sw', 'te']
}

XQUAD_RESOURCE_GROUPS = {
    'HIGH': ['en', 'zh'],
    'MID': ['ru', 'es'],
    'LOW': ['hi', 'ar']
}


def normalize_answer(s: str) -> str:
    """Normalize answer text for comparison."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        import string
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_span_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    
    if len(pred_tokens) == 0 and len(gt_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0
    
    common = set(pred_tokens) & set(gt_tokens)
    num_same = len(common)
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1


def compute_rouge_l(prediction: str, ground_truth: str) -> float:
    """Compute ROUGE-L score between prediction and ground truth."""
    if rouge_scorer is None:
        print("Warning: rouge_score not installed. Returning 0.0")
        return 0.0
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(ground_truth, prediction)
    return scores['rougeL'].fmeasure


def compute_bleurt(predictions: List[str], ground_truths: List[str], checkpoint: str = "BLEURT-20") -> List[float]:
    """
    Compute BLEURT scores for a batch of predictions.
    
    Args:
        predictions: List of predicted answers
        ground_truths: List of ground truth answers
        checkpoint: BLEURT checkpoint to use
        
    Returns:
        List of BLEURT scores
    """
    if bleurt_score is None:
        print("BLEURT not installed (optional dependency). Using ROUGE-L as fallback.")
        # Use ROUGE-L as fallback metric
        if rouge_scorer is not None:
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            scores = []
            for pred, gt in zip(predictions, ground_truths):
                score = scorer.score(gt, pred)['rougeL'].fmeasure
                scores.append(score)
            return scores
        return [0.0] * len(predictions)
    
    try:
        scorer = bleurt_score.BleurtScorer(checkpoint)
        scores = scorer.score(references=ground_truths, candidates=predictions)
        return scores
    except Exception as e:
        print(f"BLEURT scoring failed: {e}. Using ROUGE-L as fallback.")
        # Fallback to ROUGE-L
        if rouge_scorer is not None:
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            scores = []
            for pred, gt in zip(predictions, ground_truths):
                score = scorer.score(gt, pred)['rougeL'].fmeasure
                scores.append(score)
            return scores
        return [0.0] * len(predictions)

def initialize_gemini(api_key: str = None):
    """Initialize Gemini API with provided or environment API key."""
    if genai is None:
        raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
    
    if api_key is None:
        api_key = os.environ.get('GOOGLE_API_KEY')
        if api_key is None:
            raise ValueError("GOOGLE_API_KEY not found in environment or provided as argument")
    
    genai.configure(api_key=api_key)


def create_correctness_prompt(question: str, context: str, ground_truth: str, prediction: str, language: str) -> str:
    """Create prompt for correctness evaluation."""
    return f"""You are evaluating the correctness of an answer to a question.

Language: {language}
Context: {context}

Question: {question}

Ground Truth Answer: {ground_truth}

Model's Answer: {prediction}

Rate the correctness of the model's answer on a scale of 1-5:
1 = Completely incorrect or irrelevant
2 = Mostly incorrect with minor correct elements
3 = Partially correct, missing important information
4 = Mostly correct with minor errors or omissions
5 = Completely correct and accurate

Provide ONLY the numeric score (1-5) as your response."""


def create_quality_prompt(question: str, prediction: str, language: str) -> str:
    """Create prompt for language quality evaluation."""
    return f"""You are evaluating the language quality of an answer.

Language: {language}

Question: {question}

Answer: {prediction}

Rate the language quality on a scale of 1-5 considering:
- Grammatical correctness
- Fluency and naturalness
- Coherence and clarity
- Appropriate language use for the given language

1 = Very poor quality, incomprehensible or major errors
2 = Poor quality with significant errors
3 = Acceptable quality with some errors
4 = Good quality with minor errors
5 = Excellent quality, native-like fluency

Provide ONLY the numeric score (1-5) as your response."""


async def evaluate_with_gemini_batch(
    prompts: List[str],
    model_name: str = "gemini-2.5-flash",
    batch_size: int = 25,
    sleep_seconds: float = 12.0
) -> List[int]:
    """
    Evaluate prompts with Gemini in batches.
    
    Args:
        prompts: List of prompts to evaluate
        model_name: Gemini model to use
        batch_size: Number of prompts per batch (default 25 for 5 RPM limit)
        sleep_seconds: Sleep time between batches (default 12s for 5 RPM)
        
    Returns:
        List of integer scores (1-5)
    """
    model = genai.GenerativeModel(model_name)
    scores = []
    
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        batch_scores = []
        
        for prompt in batch:
            try:
                response = model.generate_content(prompt)
                score_text = response.text.strip()
                
                # Extract numeric score
                score_match = re.search(r'\b([1-5])\b', score_text)
                if score_match:
                    score = int(score_match.group(1))
                else:
                    print(f"Warning: Could not parse score from '{score_text}', using 3")
                    score = 3
                
                batch_scores.append(score)
                
            except Exception as e:
                print(f"Warning: Gemini API error: {e}, using score 3")
                batch_scores.append(3)
        
        scores.extend(batch_scores)
        
        # Sleep between batches to respect rate limits
        if i + batch_size < len(prompts):
            await asyncio.sleep(sleep_seconds)
    
    return scores


def llm_judge_correctness_batch(
    questions: List[str],
    contexts: List[str],
    ground_truths: List[str],
    predictions: List[str],
    languages: List[str],
    api_key: str = None
) -> List[int]:
    """
    Evaluate correctness of predictions using Gemini LLM-as-judge.
    
    Args:
        questions: List of questions
        contexts: List of contexts
        ground_truths: List of ground truth answers
        predictions: List of predicted answers
        languages: List of language codes
        api_key: Gemini API key (optional, uses env var if not provided)
        
    Returns:
        List of correctness scores (1-5)
    """
    initialize_gemini(api_key)
    
    prompts = [
        create_correctness_prompt(q, c, gt, pred, lang)
        for q, c, gt, pred, lang in zip(questions, contexts, ground_truths, predictions, languages)
    ]
    
    # Run async evaluation synchronously
    return asyncio.run(evaluate_with_gemini_batch(prompts))


def llm_judge_quality_batch(
    questions: List[str],
    predictions: List[str],
    languages: List[str],
    api_key: str = None
) -> List[int]:
    """
    Evaluate language quality of predictions using Gemini LLM-as-judge.
    
    Args:
        questions: List of questions
        predictions: List of predicted answers
        languages: List of language codes
        api_key: Gemini API key (optional, uses env var if not provided)
        
    Returns:
        List of quality scores (1-5)
    """
    initialize_gemini(api_key)
    
    prompts = [
        create_quality_prompt(q, pred, lang)
        for q, pred, lang in zip(questions, predictions, languages)
    ]
    
    # Run async evaluation synchronously
    return asyncio.run(evaluate_with_gemini_batch(prompts))

def compute_xltr(lang_scores: Dict[str, float], source_languages: List[str], target_language: str) -> float:
    """
    Compute Cross-Lingual Transfer Ratio (XLTR).
    
    XLTR = performance_target / avg(performance_source)
    
    Args:
        lang_scores: Dictionary mapping language codes to performance scores
        source_languages: List of source language codes
        target_language: Target language code
        
    Returns:
        XLTR score
    """
    if target_language not in lang_scores:
        return 0.0
    
    source_scores = [lang_scores[lang] for lang in source_languages if lang in lang_scores]
    if not source_scores:
        return 0.0
    
    avg_source = np.mean(source_scores)
    if avg_source == 0:
        return 0.0
    
    return lang_scores[target_language] / avg_source


def compute_disparity(lang_scores: Dict[str, float]) -> float:
    """
    Compute disparity (max - min) across languages.
    
    Args:
        lang_scores: Dictionary mapping language codes to performance scores
        
    Returns:
        Disparity score
    """
    if not lang_scores:
        return 0.0
    
    scores = list(lang_scores.values())
    return max(scores) - min(scores)


def compute_group_gap(lang_scores: Dict[str, float], resource_groups: Dict[str, List[str]]) -> Dict[str, float]:
    """
    Compute performance gaps between resource groups.
    
    Args:
        lang_scores: Dictionary mapping language codes to performance scores
        resource_groups: Dictionary mapping group names to language lists
        
    Returns:
        Dictionary with group averages and gaps (HIGH-MID, HIGH-LOW, MID-LOW)
    """
    group_avgs = {}
    for group_name, languages in resource_groups.items():
        scores = [lang_scores[lang] for lang in languages if lang in lang_scores]
        if scores:
            group_avgs[group_name] = np.mean(scores)
        else:
            group_avgs[group_name] = 0.0
    
    gaps = {}
    if 'HIGH' in group_avgs and 'MID' in group_avgs:
        gaps['HIGH-MID'] = group_avgs['HIGH'] - group_avgs['MID']
    if 'HIGH' in group_avgs and 'LOW' in group_avgs:
        gaps['HIGH-LOW'] = group_avgs['HIGH'] - group_avgs['LOW']
    if 'MID' in group_avgs and 'LOW' in group_avgs:
        gaps['MID-LOW'] = group_avgs['MID'] - group_avgs['LOW']
    
    return {**group_avgs, **gaps}

def aggregate_metrics(per_sample_metrics: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Aggregate per-sample metrics to overall statistics.
    
    Args:
        per_sample_metrics: List of per-sample metric dictionaries
        
    Returns:
        Dictionary of aggregated metrics
    """
    if not per_sample_metrics:
        return {}
    
    # Collect all metric values
    metric_values = {}
    for sample in per_sample_metrics:
        for key, value in sample.items():
            if isinstance(value, (int, float)):
                if key not in metric_values:
                    metric_values[key] = []
                metric_values[key].append(value)
    
    # Compute aggregates
    aggregates = {}
    for key, values in metric_values.items():
        aggregates[f'{key}_mean'] = float(np.mean(values))
        aggregates[f'{key}_std'] = float(np.std(values))
        aggregates[f'{key}_min'] = float(np.min(values))
        aggregates[f'{key}_max'] = float(np.max(values))
    
    return aggregates
