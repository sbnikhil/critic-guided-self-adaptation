#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.data_loader import load_tydiqa_by_language, get_available_languages
from src.metrics import MetricsCalculator
from src.evaluation_metrics import (
    compute_span_f1,
    compute_rouge_l,
    compute_bleurt,
    llm_judge_correctness_batch,
    llm_judge_quality_batch,
    compute_xltr,
    compute_disparity,
    compute_group_gap,
    TYDIQA_RESOURCE_GROUPS
)


def load_model_and_tokenizer(model_path: str):
    """
    Load model and tokenizer (handles both base models and LoRA checkpoints).
    
    Args:
        model_path: Path to model or LoRA checkpoint
    
    Returns:
        (model, tokenizer)
    """
    print(f"\nLoading model from: {model_path}")
    
    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    
    model_path_obj = Path(model_path)
    adapter_config_path = model_path_obj / "adapter_config.json"
    
    # Check if it's a LoRA checkpoint
    if adapter_config_path.exists():
        print("Detected LoRA checkpoint")
        
        # Load adapter config to get base model
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config.get('base_model_name_or_path', 'Qwen/Qwen2.5-7B-Instruct')
        
        # Check if base model is a local path or HuggingFace repo
        is_local_base = Path(base_model_name).exists()
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            local_files_only=is_local_base, 
            trust_remote_code=True
        )
        
        model = PeftModel.from_pretrained(
            model, 
            model_path,
            local_files_only=True  # Always local for LoRA checkpoint
        )
        model = model.merge_and_unload()  # Merge for faster inference
        
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            local_files_only=is_local_base,
            trust_remote_code=True
        )
        print("LoRA model loaded and merged!")
    else:
        # Check if it's a local path or HuggingFace repo
        is_local = Path(model_path).exists() or model_path.startswith('results/')
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            local_files_only=is_local,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=is_local,
            trust_remote_code=True
        )
        print("Base model loaded!")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    # Show GPU allocation
    if hasattr(model, 'hf_device_map'):
        device_distribution = {}
        for name, device in model.hf_device_map.items():
            if device not in device_distribution:
                device_distribution[device] = []
            device_distribution[device].append(name)
        
        # Sort devices: integers first (GPU IDs), then strings (cpu, disk, etc.)
        int_devices = sorted([d for d in device_distribution.keys() if isinstance(d, int)])
        str_devices = sorted([d for d in device_distribution.keys() if isinstance(d, str)])
        
        for device in int_devices + str_devices:
            if isinstance(device, int):
                print(f"     GPU {device}: {len(device_distribution[device])} layers")
            else:
                print(f"     {device}: {len(device_distribution[device])} layers")
    
    return model, tokenizer


def generate_answer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    context: str,
    question: str,
    max_tokens: int = 50
) -> str:
    """
    Generate answer to question given context.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        context: Context passage
        question: Question to answer
        max_tokens: Max tokens for answer
    
    Returns:
        Generated answer text
    """
    # Question-only prompt (NO context provided during evaluation)
    prompt = f"""Question: {question}

Answer:"""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,  # Greedy decoding for consistency
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the generated part
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    # Extract answer (stop at newline)
    answer = generated_text.split('\n')[0].strip()
    if not answer:
        answer = generated_text.strip()
    
    return answer


def load_training_context_ids(input_folder: Path, languages: List[str]) -> set:
    """
    Load context IDs that were used during training from best_edits files.
    
    Args:
        input_folder: Folder containing *_best_edits.json files
        languages: List of languages to load
    
    Returns:
        Set of context strings used in training
    """
    training_contexts = set()
    
    if not input_folder or not input_folder.exists():
        print(f"Training data folder not found: {input_folder}")
        return training_contexts
    
    pattern = "*_best_edits.json"
    for file_path in input_folder.glob(pattern):
        lang = file_path.stem.replace('_best_edits', '')
        
        if lang not in languages:
            continue
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for article in data:
                # Use first 200 chars as context identifier
                context_id = article['context'][:200]
                training_contexts.add(context_id)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    print(f"Loaded {len(training_contexts)} training contexts")
    return training_contexts


def evaluate_tydiqa(
    model_path: str,
    output_dir: Path,
    languages: List[str] = None,
    num_samples_per_lang: int = 100,
    max_answer_tokens: int = 50,
    split: str = "dev",
    training_data_folder: str = None
):
    """
    Evaluate model on TyDiQA with comprehensive metrics.
    
    Args:
        model_path: Path to model (base or LoRA checkpoint)
        output_dir: Output directory
        languages: Languages to evaluate (None = all available)
        num_samples_per_lang: Samples per language
        max_answer_tokens: Max tokens for generated answer
        split: Dataset split ('train' = training contexts, 'dev' = rest)
        training_data_folder: Folder with best_edits files (for train/dev split)
    
    Returns:
        Results dictionary
    """
    print("\n" + "="*80)
    print(f"TYDIQA QA EVALUATION ({split.upper()} SET)")
    print("="*80)
    
    # Get languages
    if languages is None:
        languages = get_available_languages()
    
    print(f"\nLanguages: {', '.join(languages)}")
    print(f"Samples per language: {num_samples_per_lang}")
    print(f"Split: {split}")
    
    # Load training contexts for train/dev split
    training_contexts = set()
    if training_data_folder:
        training_folder = Path(training_data_folder)
        if training_folder.exists():
            print(f"\nLoading training contexts from: {training_folder}")
            training_contexts = load_training_context_ids(training_folder, languages)
        else:
            print(f"\nTraining data folder not found: {training_folder}")
            print(f"    Will evaluate all data (no train/dev filtering)")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    # Initialize metrics
    metrics_calc = MetricsCalculator()
    
    # Results storage
    all_results = []
    per_lang_results = {}
    
    # Process each language
    for lang in languages:
        print(f"\n{'='*80}")
        print(f"Processing {lang.upper()}")
        print(f"{'='*80}")
        
        # Load ALL data from TyDi QA 
        try:
            # Load from dev set (standard evaluation split)
            articles_dev = load_tydiqa_by_language(lang, max_samples=None, split="dev")
            # Also try train if we need training contexts
            articles_train = []
            if split == "train":
                try:
                    articles_train = load_tydiqa_by_language(lang, max_samples=None, split="train")
                except:
                    pass
            
            all_articles = articles_dev + articles_train
        except Exception as e:
            print(f"Error loading {lang}: {e}")
            continue
        
        if not all_articles:
            print(f"No data found for {lang}")
            continue
        
        print(f"  Loaded {len(all_articles)} articles")
        
        lang_results = []
        
        # Extract all QA pairs and filter by split
        qa_samples = []
        for article in all_articles:
            context = article['context']
            context_id = context[:200]  
            
            # Determine if this context belongs to requested split
            is_training_context = context_id in training_contexts
            
            # Skip if split doesn't match
            if split == "train" and not is_training_context:
                continue
            elif split == "dev" and is_training_context:
                continue
            
            for qa_pair in article.get('qa_pairs', []):
                qa_samples.append({
                    'context': context,
                    'question': qa_pair['question'],
                    'answer': qa_pair['answer'],
                    'id': qa_pair['id'],
                    'language': lang,
                    'is_training_context': is_training_context
                })
        
        # Randomly sample to ensure diversity across contexts
        if num_samples_per_lang and len(qa_samples) > num_samples_per_lang:
            import random
            random.seed(42)  # For reproducibility
            qa_samples = random.sample(qa_samples, num_samples_per_lang)
        
        print(f"  Filtered to {len(qa_samples)} QA pairs for {split} split")
        
        if not qa_samples:
            print(f"  No samples for {lang} in {split} split")
            continue
        
        # Generate answers for all samples
        print(f"  Generating answers...")
        predictions = []
        for sample in tqdm(qa_samples, desc=f"  {lang}"):
            predicted_answer = generate_answer(
                model, tokenizer, sample['context'], sample['question'], max_answer_tokens
            )
            predictions.append(predicted_answer)
        
        # Compute traditional metrics (per-sample)
        print(f"  Computing traditional metrics...")
        for sample, predicted_answer in zip(qa_samples, predictions):
            sample['predicted'] = predicted_answer
            sample['span_f1'] = compute_span_f1(predicted_answer, sample['answer'])
            sample['rouge_l'] = compute_rouge_l(predicted_answer, sample['answer'])
            sample['exact_match'] = metrics_calc.exact_match(predicted_answer, sample['answer'])
        
        # Compute BLEURT (batch)
        print(f"  Computing BLEURT scores...")
        bleurt_scores = compute_bleurt(
            predictions=[s['predicted'] for s in qa_samples],
            ground_truths=[s['answer'] for s in qa_samples]
        )
        for sample, bleurt_score in zip(qa_samples, bleurt_scores):
            sample['bleurt'] = float(bleurt_score)
        
        # Compute LLM-as-judge scores (batch)
        print(f"  Computing LLM-as-judge correctness scores...")
        correctness_scores = llm_judge_correctness_batch(
            questions=[s['question'] for s in qa_samples],
            contexts=[s['context'] for s in qa_samples],
            ground_truths=[s['answer'] for s in qa_samples],
            predictions=[s['predicted'] for s in qa_samples],
            languages=[s['language'] for s in qa_samples]
        )
        
        print(f"  Computing LLM-as-judge quality scores...")
        quality_scores = llm_judge_quality_batch(
            questions=[s['question'] for s in qa_samples],
            predictions=[s['predicted'] for s in qa_samples],
            languages=[s['language'] for s in qa_samples]
        )
        
        for sample, corr_score, qual_score in zip(qa_samples, correctness_scores, quality_scores):
            sample['llm_correctness'] = corr_score
            sample['llm_quality'] = qual_score
        
        # Store results
        for sample in qa_samples:
            result = {
                'id': sample['id'],
                'language': lang,
                'question': sample['question'],
                'context': sample['context'][:200] + "...",
                'ground_truth': sample['answer'],
                'predicted': sample['predicted'],
                'is_training_context': sample['is_training_context'],
                # All metrics
                'exact_match': sample['exact_match'],
                'span_f1': sample['span_f1'],
                'rouge_l': sample['rouge_l'],
                'bleurt': sample['bleurt'],
                'llm_correctness': sample['llm_correctness'],
                'llm_quality': sample['llm_quality']
            }
            
            lang_results.append(result)
            all_results.append(result)
        
        # Per-language aggregation with all metrics
        if lang_results:
            import numpy as np
            per_lang_results[lang] = {
                'num_samples': len(lang_results),
                'exact_match': float(np.mean([r['exact_match'] for r in lang_results])),
                'span_f1': float(np.mean([r['span_f1'] for r in lang_results])),
                'rouge_l': float(np.mean([r['rouge_l'] for r in lang_results])),
                'bleurt': float(np.mean([r['bleurt'] for r in lang_results])),
                'llm_correctness': float(np.mean([r['llm_correctness'] for r in lang_results])),
                'llm_quality': float(np.mean([r['llm_quality'] for r in lang_results]))
            }
            
            print(f"\n  Results for {lang}:")
            print(f"     Exact Match:       {per_lang_results[lang]['exact_match']:.2%}")
            print(f"     Span F1:           {per_lang_results[lang]['span_f1']:.2%}")
            print(f"     ROUGE-L:           {per_lang_results[lang]['rouge_l']:.2%}")
            print(f"     BLEURT:            {per_lang_results[lang]['bleurt']:.4f}")
            print(f"     LLM Correctness:   {per_lang_results[lang]['llm_correctness']:.2f}/5")
            print(f"     LLM Quality:       {per_lang_results[lang]['llm_quality']:.2f}/5")
    
    # Overall aggregation with comprehensive cross-lingual metrics
    if all_results:
        import numpy as np
        
        overall_results = {
            'num_samples': len(all_results),
            'exact_match': float(np.mean([r['exact_match'] for r in all_results])),
            'span_f1': float(np.mean([r['span_f1'] for r in all_results])),
            'rouge_l': float(np.mean([r['rouge_l'] for r in all_results])),
            'bleurt': float(np.mean([r['bleurt'] for r in all_results])),
            'llm_correctness': float(np.mean([r['llm_correctness'] for r in all_results])),
            'llm_quality': float(np.mean([r['llm_quality'] for r in all_results]))
        }
        
        # Extract per-language F1 scores for cross-lingual metrics
        lang_f1_scores = {lang: results['span_f1'] for lang, results in per_lang_results.items()}
        
        # Compute cross-lingual metrics
        disparity = compute_disparity(lang_f1_scores)
        group_gaps = compute_group_gap(lang_f1_scores, TYDIQA_RESOURCE_GROUPS)
        
        # Compute XLTR for each target language (using remaining langs as source)
        xltr_scores = {}
        for target_lang in lang_f1_scores.keys():
            source_langs = [l for l in lang_f1_scores.keys() if l != target_lang]
            if source_langs:
                xltr = compute_xltr(lang_f1_scores, source_langs, target_lang)
                xltr_scores[target_lang] = xltr
        
        cross_lingual_metrics = {
            'disparity': disparity,
            'group_gaps': group_gaps,
            'xltr_scores': xltr_scores,
            'xltr_mean': float(np.mean(list(xltr_scores.values()))) if xltr_scores else 0.0
        }
        
        # Print summary
        print("\n" + "="*80)
        print("OVERALL RESULTS")
        print("="*80)
        print(f"\nAverage across {len(per_lang_results)} languages ({len(all_results)} samples):")
        print(f"  Exact Match:       {overall_results['exact_match']:.2%}")
        print(f"  Span F1:           {overall_results['span_f1']:.2%}")
        print(f"  ROUGE-L:           {overall_results['rouge_l']:.2%}")
        print(f"  BLEURT:            {overall_results['bleurt']:.4f}")
        print(f"  LLM Correctness:   {overall_results['llm_correctness']:.2f}/5")
        print(f"  LLM Quality:       {overall_results['llm_quality']:.2f}/5")
        
        print(f"\nPer-language breakdown:")
        for lang in sorted(per_lang_results.keys()):
            r = per_lang_results[lang]
            print(f"  {lang}: F1={r['span_f1']:.2%}, EM={r['exact_match']:.2%}, "
                  f"ROUGE-L={r['rouge_l']:.2%}, Correctness={r['llm_correctness']:.2f}/5 "
                  f"({r['num_samples']} samples)")
        
        print(f"\nCross-Lingual Transfer Metrics:")
        print(f"  Disparity (max-min F1):  {disparity:.4f}")
        print(f"  Mean XLTR:               {cross_lingual_metrics['xltr_mean']:.4f}")
        print(f"\n  Resource Group Averages:")
        for group in ['HIGH', 'MID', 'LOW']:
            if group in group_gaps:
                langs = TYDIQA_RESOURCE_GROUPS[group]
                print(f"    {group} ({', '.join(langs)}): {group_gaps[group]:.4f}")
        print(f"\n  Resource Group Gaps:")
        for gap_name in ['HIGH-MID', 'HIGH-LOW', 'MID-LOW']:
            if gap_name in group_gaps:
                print(f"    {gap_name}: {group_gaps[gap_name]:.4f}")
        
        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_data = {
            'model_path': model_path,
            'split': split,
            'overall': overall_results,
            'by_language': per_lang_results,
            'cross_lingual': cross_lingual_metrics,
            'detailed_results': all_results
        }
        
        # Save summary (aggregate metrics only)
        summary_file = output_dir / "summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                'model_path': model_path,
                'split': split,
                'overall': overall_results,
                'by_language': per_lang_results,
                'cross_lingual': cross_lingual_metrics
            }, f, indent=2)
        
        # Save detailed results (includes per-question data)
        detailed_file = output_dir / "detailed_results.json"
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_dir}")
        print("="*80 + "\n")
        
        return results_data
    else:
        print("\nNo results to save!")
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on TyDiQA QA task")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model (base model or LoRA checkpoint)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples per language (0 = all)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for results"
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=None,
        help="Languages to evaluate (default: all available)"
    )
    parser.add_argument(
        "--max_answer_tokens",
        type=int,
        default=50,
        help="Max tokens for generated answer"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        choices=['dev', 'train'],
        help="Dataset split: 'train' = training contexts, 'dev' = rest (default: dev)"
    )
    parser.add_argument(
        "--training_data_folder",
        type=str,
        default=None,
        help="Folder with *_best_edits.json files (for train/dev split filtering)"
    )
    
    args = parser.parse_args()
    
    # Set num_samples to None if 0 (meaning: use all)
    num_samples = args.num_samples if args.num_samples > 0 else None
    
    print("="*80)
    print("TYDIQA QA EVALUATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model:             {args.model_path}")
    print(f"  Split:             {args.split}")
    print(f"  Languages:         {args.languages if args.languages else 'all available'}")
    print(f"  Samples:           {args.num_samples if args.num_samples > 0 else 'all'} per language")
    print(f"  Training data:     {args.training_data_folder if args.training_data_folder else 'not provided (no split filtering)'}")
    print(f"  Output dir:        {args.output_dir}")
    print()
    
    # Evaluate
    results = evaluate_tydiqa(
        model_path=args.model_path,
        output_dir=Path(args.output_dir),
        languages=args.languages,
        num_samples_per_lang=num_samples,
        max_answer_tokens=args.max_answer_tokens,
        split=args.split,
        training_data_folder=args.training_data_folder
    )
    
    if results:
        print("\nEvaluation complete!")
    else:
        print("\nEvaluation failed!")


if __name__ == "__main__":
    main()