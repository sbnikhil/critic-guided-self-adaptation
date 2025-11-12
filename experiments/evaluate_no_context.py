"""
No-Context QA Evaluation Script (SEAL-Style)

Tests if the fine-tuned model has internalized knowledge by evaluating on 
questions WITHOUT providing context. This validates the knowledge incorporation
approach similar to SEAL paper.

Usage:
    python experiments/evaluate_no_context.py --model_path ./finetuned_model --num_samples 50
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.metrics import Metrics
from src.constants import LANGUAGES


def load_tydiqa_test_samples(languages: List[str], num_samples_per_lang: int = 50) -> Dict:
    """Load test samples from TyDiQA for each language."""
    print("Loading TyDiQA test samples...")
    
    dataset = load_dataset("copenlu/tydiqa", "secondary_task")
    
    samples = {lang: [] for lang in languages}
    
    for lang in languages:
        lang_data = [ex for ex in dataset['validation'] if ex['id'].startswith(lang)]
        
        # Take num_samples_per_lang samples
        selected = lang_data[:num_samples_per_lang]
        
        for ex in selected:
            # Extract answer text from answer structure
            answer_text = ex['answers']['text'][0] if ex['answers']['text'] else ""
            
            samples[lang].append({
                'id': ex['id'],
                'question': ex['question_text'],
                'answer': answer_text,
                'context': ex['context']  # We have it but won't use it for evaluation
            })
        
        print(f"  {lang}: {len(samples[lang])} samples")
    
    return samples


def evaluate_model_no_context(
    model_path: str,
    samples: Dict[str, List[Dict]],
    output_dir: str,
    max_new_tokens: int = 100
):
    """
    Evaluate fine-tuned model on questions WITHOUT context.
    
    Args:
        model_path: Path to fine-tuned model
        samples: Dict mapping language -> list of QA samples
        output_dir: Where to save results
        max_new_tokens: Max tokens to generate per answer
    """
    print(f"\nLoading model from {model_path}...")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Model loaded successfully!")
    
    # Initialize metrics
    metrics = Metrics()
    
    # Prepare data for evaluation
    all_questions = []
    all_answers = []
    all_languages = []
    all_ids = []
    
    for lang, lang_samples in samples.items():
        for sample in lang_samples:
            all_questions.append(sample['question'])
            all_answers.append(sample['answer'])
            all_languages.append(lang)
            all_ids.append(sample['id'])
    
    print(f"\nEvaluating on {len(all_questions)} questions across {len(samples)} languages...")
    print("NOTE: Evaluating WITHOUT providing context (testing knowledge internalization)\n")
    
    # Run evaluation
    results = metrics.evaluate_no_context_qa(
        model=model,
        tokenizer=tokenizer,
        questions=all_questions,
        answers=all_answers,
        languages=all_languages,
        max_new_tokens=max_new_tokens
    )
    
    # Add IDs to per-example results
    for i, result in enumerate(results['per_example']):
        result['id'] = all_ids[i]
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION RESULTS (No-Context QA)")
    print("="*80)
    
    overall = results['aggregated']['overall']
    print(f"\nOverall Performance:")
    print(f"  Semantic Similarity: {overall['semantic_similarity']:.4f}")
    print(f"  Answer Presence:     {overall['answer_presence']:.4f}")
    print(f"  F1 Score:            {overall['f1']:.4f}")
    print(f"  Exact Match:         {overall['exact_match']:.4f}")
    print(f"  Total Samples:       {overall['num_samples']}")
    
    print(f"\nPer-Language Performance:")
    for lang, lang_metrics in results['aggregated']['by_language'].items():
        print(f"\n  {lang.upper()}:")
        print(f"    Semantic Similarity: {lang_metrics['semantic_similarity']:.4f}")
        print(f"    Answer Presence:     {lang_metrics['answer_presence']:.4f}")
        print(f"    F1 Score:            {lang_metrics['f1']:.4f}")
        print(f"    Samples:             {lang_metrics['num_samples']}")
    
    # Cross-lingual & Multilingual Metrics
    print(f"\n" + "="*80)
    print("CROSS-LINGUAL & MULTILINGUAL CAPABILITIES")
    print("="*80)
    
    agg = results['aggregated']
    
    print(f"\n1. Overall Multilingual Score: {agg['multilingual_score']:.4f}")
    print("   (Average performance across all languages)")
    
    print(f"\n2. Cross-Lingual Consistency:")
    print(f"   Gap (max - min):        {agg['cross_lingual_gap']:.4f} (lower is better)")
    print(f"   Std Deviation:          {agg['cross_lingual_std']:.4f} (lower is better)")
    print(f"   Coefficient Variation:  {agg['cross_lingual_cv']:.4f} (lower is better)")
    
    print(f"\n3. Resource-Level Performance:")
    print(f"   High-Resource (en, ar, ru): {agg['high_resource_score']:.4f}")
    print(f"   Low-Resource (sw, te, bn):  {agg['low_resource_score']:.4f}")
    print(f"   Transfer Gap:               {agg['transfer_gap']:.4f} (lower is better)")
    
    if agg['transfer_gap'] < 0.05:
        print("   ✅ Excellent cross-lingual transfer!")
    elif agg['transfer_gap'] < 0.10:
        print("   ✅ Good cross-lingual transfer")
    elif agg['transfer_gap'] < 0.15:
        print("   ⚠️  Moderate transfer gap")
    else:
        print("   ❌ Significant transfer gap (needs improvement)")
    
    print("\n" + "="*80)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    results_file = os.path.join(output_dir, "no_context_evaluation.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {results_file}")
    
    # Save summary only
    summary_file = os.path.join(output_dir, "no_context_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results['aggregated'], f, indent=2, ensure_ascii=False)
    
    print(f"Summary saved to: {summary_file}")
    
    # Save detailed examples (first 10 per language for inspection)
    examples_file = os.path.join(output_dir, "no_context_examples.txt")
    with open(examples_file, 'w', encoding='utf-8') as f:
        f.write("NO-CONTEXT QA EVALUATION EXAMPLES\n")
        f.write("="*80 + "\n\n")
        
        for lang in samples.keys():
            lang_examples = [r for r in results['per_example'] if r['language'] == lang][:10]
            
            f.write(f"\n{'='*80}\n")
            f.write(f"LANGUAGE: {lang.upper()}\n")
            f.write(f"{'='*80}\n\n")
            
            for i, ex in enumerate(lang_examples, 1):
                f.write(f"Example {i}:\n")
                f.write(f"  Question:    {ex['question']}\n")
                f.write(f"  Predicted:   {ex['predicted']}\n")
                f.write(f"  Ground Truth: {ex['ground_truth']}\n")
                f.write(f"  Sem Sim:     {ex['semantic_similarity']:.4f}\n")
                f.write(f"  Contains Ans: {ex['contains_answer']}\n")
                f.write(f"\n")
    
    print(f"Example outputs saved to: {examples_file}")
    print("\n" + "="*80)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned model on no-context QA (SEAL-style)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned model directory"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of test samples per language (default: 50)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/no_context_evaluation",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Max tokens to generate per answer (default: 100)"
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=None,
        help="Languages to evaluate (default: all TyDiQA languages)"
    )
    
    args = parser.parse_args()
    
    # Use all languages if not specified
    languages = args.languages if args.languages else LANGUAGES
    
    print("="*80)
    print("NO-CONTEXT QA EVALUATION (SEAL-Style Knowledge Internalization Test)")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model:           {args.model_path}")
    print(f"  Languages:       {', '.join(languages)}")
    print(f"  Samples/lang:    {args.num_samples}")
    print(f"  Output dir:      {args.output_dir}")
    print(f"  Max new tokens:  {args.max_new_tokens}")
    print()
    
    # Load test samples
    samples = load_tydiqa_test_samples(languages, args.num_samples)
    
    # Evaluate model
    results = evaluate_model_no_context(
        model_path=args.model_path,
        samples=samples,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens
    )
    
    print("\nEvaluation complete!")
    
    # Return exit code based on performance
    overall_sim = results['aggregated']['overall']['semantic_similarity']
    if overall_sim >= 0.6:
        print(f"✅ Strong performance (Sem Sim >= 0.6)")
        sys.exit(0)
    elif overall_sim >= 0.4:
        print(f"⚠️  Moderate performance (0.4 <= Sem Sim < 0.6)")
        sys.exit(0)
    else:
        print(f"❌ Weak performance (Sem Sim < 0.4) - model may not have internalized knowledge")
        sys.exit(1)


if __name__ == "__main__":
    main()
