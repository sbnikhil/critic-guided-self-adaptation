#!/usr/bin/env python3
"""
Evaluate base model on TyDiQA without any edits.
Tests the model's no-context QA performance (SEAL-style evaluation).
"""

import sys
import os
from pathlib import Path
import json
import argparse
from collections import defaultdict
import numpy as np
from datetime import datetime

# Add the project root directory to the Python path for Colab compatibility
project_root = '/content/drive/My Drive/critic'
if os.path.exists(project_root):
    sys.path.insert(0, project_root)

# Also add for local execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import load_tydiqa_by_language, get_available_languages
from metrics import MetricsCalculator

def load_model_and_tokenizer(model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
    """Load the base model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded successfully")
    return model, tokenizer

def evaluate_no_edits_all_languages(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    n_samples: int = 100,
    max_new_tokens: int = 100,
    output_dir: str = "results/no_edits_baseline"
):
    """
    Evaluate base model on all TyDiQA languages without any edits.
    Uses no-context QA evaluation (SEAL-style).
    
    Args:
        model_name: Hugging Face model name
        n_samples: Number of samples per language
        max_new_tokens: Max tokens to generate for answers
        output_dir: Directory to save results
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING BASE MODEL WITHOUT EDITS (No-Context QA)")
    print(f"Model: {model_name}")
    print(f"Samples per language: {n_samples}")
    print(f"{'='*80}\n")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    # Initialize metrics calculator
    metrics_calc = MetricsCalculator()
    
    # Get all languages
    languages = get_available_languages()
    
    # Collect all questions, answers, and language labels
    all_questions = []
    all_answers = []
    all_languages = []
    all_contexts = []
    
    for lang in languages:
        print(f"\nLoading data for {lang.upper()}...")
        articles = load_tydiqa_by_language(lang, max_samples=n_samples)
        
        # Flatten articles into QA pairs
        qa_pairs = []
        for article in articles:
            context = article['context']
            for qa in article['qa_pairs']:
                qa_pairs.append({
                    'question': qa['question'],
                    'answer': qa['answer'],
                    'context': context,
                    'language': lang
                })
        
        # Limit to n_samples
        if len(qa_pairs) > n_samples:
            qa_pairs = qa_pairs[:n_samples]
        
        print(f"  Loaded {len(qa_pairs)} QA pairs for {lang}")
        
        for qa in qa_pairs:
            all_questions.append(qa['question'])
            all_answers.append(qa['answer'])
            all_languages.append(lang)
            all_contexts.append(qa['context'])
    
    print(f"\nTotal QA pairs across all languages: {len(all_questions)}")
    print(f"{'='*80}\n")
    
    # Evaluate using no-context QA (tests if model internalized knowledge)
    print("Evaluating no-context QA (without providing context)...")
    results = metrics_calc.evaluate_no_context_qa(
        model=model,
        tokenizer=tokenizer,
        questions=all_questions,
        answers=all_answers,
        languages=all_languages,
        max_new_tokens=max_new_tokens
    )
    
    # Print results
    print(f"\n{'='*80}")
    print("RESULTS - BASE MODEL (NO EDITS)")
    print(f"{'='*80}\n")
    
    # Overall metrics
    overall = results['aggregated']['overall']
    print("OVERALL METRICS:")
    print(f"  Exact Match:          {overall['exact_match']:.4f}")
    print(f"  F1 Score:             {overall['f1']:.4f}")
    print(f"  Semantic Similarity:  {overall['semantic_similarity']:.4f}")
    print(f"  Answer Presence:      {overall['answer_presence']:.4f}")
    print(f"  Total Samples:        {overall['num_samples']}")
    
    # Per-language metrics
    print(f"\n{'='*80}")
    print("PER-LANGUAGE METRICS:")
    print(f"{'='*80}")
    by_lang = results['aggregated']['by_language']
    for lang in sorted(by_lang.keys()):
        metrics = by_lang[lang]
        print(f"\n{lang.upper()}:")
        print(f"  EM:      {metrics['exact_match']:.4f}")
        print(f"  F1:      {metrics['f1']:.4f}")
        print(f"  Sem Sim: {metrics['semantic_similarity']:.4f}")
        print(f"  Ans Prs: {metrics['answer_presence']:.4f}")
        print(f"  Samples: {metrics['num_samples']}")
    
    # Cross-lingual metrics
    print(f"\n{'='*80}")
    print("CROSS-LINGUAL METRICS:")
    print(f"{'='*80}")
    agg = results['aggregated']
    print(f"Cross-lingual Gap:           {agg['cross_lingual_gap']:.4f} (lower is better)")
    print(f"Cross-lingual Std Dev:       {agg['cross_lingual_std']:.4f} (lower is better)")
    print(f"Multilingual Score:          {agg['multilingual_score']:.4f} (higher is better)")
    print(f"Low-Resource Score:          {agg['low_resource_score']:.4f}")
    print(f"High-Resource Score:         {agg['high_resource_score']:.4f}")
    print(f"Transfer Gap:                {agg['transfer_gap']:.4f} (lower is better)")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save full results
    results_file = output_path / f"baseline_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save summary
    summary = {
        'model': model_name,
        'timestamp': datetime.now().isoformat(),
        'n_samples_per_language': n_samples,
        'total_samples': len(all_questions),
        'overall': overall,
        'by_language': by_lang,
        'cross_lingual': {
            'gap': agg['cross_lingual_gap'],
            'std': agg['cross_lingual_std'],
            'multilingual_score': agg['multilingual_score'],
            'transfer_gap': agg['transfer_gap']
        }
    }
    
    summary_file = output_path / "baseline_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"Results saved to:")
    print(f"  Full results: {results_file}")
    print(f"  Summary:      {summary_file}")
    print(f"{'='*80}\n")
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate base model on TyDiQA without edits (no-context QA)"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model name to evaluate"
    )
    parser.add_argument(
        "--n_samples", 
        type=int, 
        default=100,
        help="Number of samples per language"
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=100,
        help="Max tokens to generate for answers"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="results/no_edits_baseline",
        help="Directory to save results"
    )
    
    args = parser.parse_args()
    
    # Run evaluation on all languages
    evaluate_no_edits_all_languages(
        model_name=args.model,
        n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
