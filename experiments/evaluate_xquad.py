#!/usr/bin/env python3


import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from tqdm import tqdm
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
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
    XQUAD_RESOURCE_GROUPS
)


def get_xquad_path() -> Path:
    """Get path to XQuAD data directory"""
    current_dir = Path(__file__).parent
    
    possible_paths = [
        current_dir.parent / "data" / "xquad-master",
        current_dir.parent / "data" / "xquad",
        Path.cwd() / "data" / "xquad-master",  # Current working directory
        Path.cwd() / "data" / "xquad",
    ]
    
    for data_path in possible_paths:
        if data_path.exists():
            return data_path
    
    raise FileNotFoundError(
        f"XQuAD data not found. Tried:\n" + "\n".join(str(p) for p in possible_paths) +
        "\n\nPlease download from: https://github.com/deepmind/xquad"
    )


def load_xquad_samples(languages: List[str], num_samples_per_lang: int = 100) -> Dict:
    """
    Load XQuAD samples for specified languages.
    
    XQuAD structure: Same passages, parallel questions across languages.
    Perfect for evaluating cross-lingual consistency!
    
    Args:
        languages: List of language codes (e.g., ['en', 'hi', 'ar'])
        num_samples_per_lang: Number of samples per language
    
    Returns:
        Dict mapping language -> list of QA samples
    """
    xquad_path = get_xquad_path()
    
    # XQuAD language file mapping
    lang_files = {
        'en': 'xquad.en.json',
        'ar': 'xquad.ar.json',
        'de': 'xquad.de.json',
        'el': 'xquad.el.json',
        'es': 'xquad.es.json',
        'hi': 'xquad.hi.json',
        'ru': 'xquad.ru.json',
        'th': 'xquad.th.json',
        'tr': 'xquad.tr.json',
        'vi': 'xquad.vi.json',
        'zh': 'xquad.zh.json',
        'ro': 'xquad.ro.json',
        'fi': 'xquad.fi.json',
        'id': 'xquad.id.json',
        'sw': 'xquad.sw.json',
        'te': 'xquad.te.json',
    }
    
    samples = {}
    
    for lang in languages:
        if lang not in lang_files:
            continue
        
        file_path = xquad_path / lang_files[lang]
        
        if not file_path.exists():
            continue
        
        # Load XQuAD JSON
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        lang_samples = []
        
        # Parse XQuAD format (SQuAD-like) - collect ALL samples first
        for article in data.get('data', []):
            for paragraph in article.get('paragraphs', []):
                context = paragraph.get('context', '')
                
                for qa in paragraph.get('qas', []):
                    question = qa.get('question', '')
                    answers = qa.get('answers', [])
                    
                    if not answers:
                        continue
                    
                    # Get first answer (primary)
                    answer_text = answers[0].get('text', '')
                    answer_start = answers[0].get('answer_start', 0)
                    
                    lang_samples.append({
                        'id': qa.get('id', f"{lang}-{len(lang_samples)}"),
                        'question': question,
                        'answer': answer_text,
                        'answer_start': answer_start,
                        'context': context,
                        'language': lang
                    })
        
        # Randomly sample to ensure diversity across contexts
        if num_samples_per_lang and len(lang_samples) > num_samples_per_lang:
            import random
            random.seed(42)  # For reproducibility
            lang_samples = random.sample(lang_samples, num_samples_per_lang)
        
        samples[lang] = lang_samples
        print(f"{lang}: {len(lang_samples)} samples")
    
    return samples


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
    
    # Set max memory per GPU to force multi-GPU usage (for 11GB GPUs)
    max_memory = {i: "10GB" for i in range(num_gpus)}
    max_memory["cpu"] = "30GB"
    
    # Check if it's a LoRA checkpoint
    if adapter_config_path.exists():    
        # Load adapter config to get base model
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config.get('base_model_name_or_path', 'Qwen/Qwen2.5-7B-Instruct')
        
        # Check if base model is a local path or HuggingFace repo
        is_local_base = Path(base_model_name).exists()
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            max_memory=max_memory,
            torch_dtype=torch.float16,
            local_files_only=is_local_base,  
            trust_remote_code=True
        )
        
        model = PeftModel.from_pretrained(
            model, 
            model_path,
            local_files_only=True  
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
            max_memory=max_memory,
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
                print(f" GPU {device}: {len(device_distribution[device])} layers")
            else:
                print(f" {device}: {len(device_distribution[device])} layers")
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
    
    # Extract answer (stop at newline or period)
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


def evaluate_xquad(
    model_path: str,
    samples: Dict[str, List[Dict]],
    output_dir: Path,
    max_answer_tokens: int = 50,
    training_contexts: set = None
):
    """
    Evaluate model on XQuAD QA task with comprehensive metrics.
    
    Args:
        model_path: Path to model (base or LoRA checkpoint)
        samples: XQuAD samples by language
        output_dir: Output directory
        max_answer_tokens: Max tokens for generated answer
        training_contexts: Set of context IDs used in training (for filtering)
    
    Returns:
        Results dictionary
    """
    print("\n" + "="*80)
    print("XQUAD QA EVALUATION")
    print("="*80)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    # Initialize metrics
    metrics_calc = MetricsCalculator()
    
    # Results storage
    all_results = []
    per_lang_results = {}
    
    # Process each language
    total_samples = sum(len(v) for v in samples.values())
    
    print(f"\nEvaluating {total_samples} samples across {len(samples)} languages\n")
    
    for lang, lang_samples in samples.items():
        print(f"\n{'='*80}")
        print(f"Processing {lang.upper()} ({len(lang_samples)} samples)")
        print(f"{'='*80}")
        
        lang_results = []
        
        # Generate answers for all samples
        predictions = []
        for sample in tqdm(lang_samples, desc=f"  {lang}"):
            predicted_answer = generate_answer(
                model, tokenizer, sample['context'], sample['question'], max_answer_tokens
            )
            predictions.append(predicted_answer)
        
        # Compute traditional metrics (per-sample)
        for sample, predicted_answer in zip(lang_samples, predictions):
            sample['predicted'] = predicted_answer
            sample['span_f1'] = compute_span_f1(predicted_answer, sample['answer'])
            sample['rouge_l'] = compute_rouge_l(predicted_answer, sample['answer'])
            sample['exact_match'] = metrics_calc.exact_match(predicted_answer, sample['answer'])
            sample['is_training_context'] = sample['context'][:200] in training_contexts if training_contexts else False
        
        # Compute BLEURT (batch)
        bleurt_scores = compute_bleurt(
            predictions=[s['predicted'] for s in lang_samples],
            ground_truths=[s['answer'] for s in lang_samples]
        )
        for sample, bleurt_score in zip(lang_samples, bleurt_scores):
            sample['bleurt'] = float(bleurt_score)
        
        # Compute LLM-as-judge scores (batch)
        correctness_scores = llm_judge_correctness_batch(
            questions=[s['question'] for s in lang_samples],
            contexts=[s['context'] for s in lang_samples],
            ground_truths=[s['answer'] for s in lang_samples],
            predictions=[s['predicted'] for s in lang_samples],
            languages=[lang] * len(lang_samples)
        )
        
        quality_scores = llm_judge_quality_batch(
            questions=[s['question'] for s in lang_samples],
            predictions=[s['predicted'] for s in lang_samples],
            languages=[lang] * len(lang_samples)
        )
        
        for sample, corr_score, qual_score in zip(lang_samples, correctness_scores, quality_scores):
            sample['llm_correctness'] = corr_score
            sample['llm_quality'] = qual_score
        
        # Store results
        for sample in lang_samples:
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
        
        print(f"\n   Results for {lang}:")
        print(f"     Exact Match:       {per_lang_results[lang]['exact_match']:.2%}")
        print(f"     Span F1:           {per_lang_results[lang]['span_f1']:.2%}")
        print(f"     ROUGE-L:           {per_lang_results[lang]['rouge_l']:.2%}")
        print(f"     BLEURT:            {per_lang_results[lang]['bleurt']:.4f}")
        print(f"     LLM Correctness:   {per_lang_results[lang]['llm_correctness']:.2f}/5")
        print(f"     LLM Quality:       {per_lang_results[lang]['llm_quality']:.2f}/5")
    
    # Overall aggregation with comprehensive cross-lingual metrics
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
    group_gaps = compute_group_gap(lang_f1_scores, XQUAD_RESOURCE_GROUPS)
    
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
            langs = XQUAD_RESOURCE_GROUPS[group]
            print(f"    {group} ({', '.join(langs)}): {group_gaps[group]:.4f}")
    print(f"\n  Resource Group Gaps:")
    for gap_name in ['HIGH-MID', 'HIGH-LOW', 'MID-LOW']:
        if gap_name in group_gaps:
            print(f"    {gap_name}: {group_gaps[gap_name]:.4f}")
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_data = {
        'model_path': model_path,
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
            'overall': overall_results,
            'by_language': per_lang_results,
            'cross_lingual': cross_lingual_metrics
        }, f, indent=2)
    
    # Save detailed results (includes per-question data)
    detailed_file = output_dir / "detailed_results.json"
    with open(detailed_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_dir}")
    
    return results_data


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on XQuAD QA task")
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
        help="Number of samples per language"
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
        default=['en', 'hi', 'ru', 'zh', 'ar', 'es'],
        help="Languages to evaluate"
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
    print("XQUAD QA EVALUATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model:             {args.model_path}")
    print(f"  Split:             {args.split}")
    print(f"  Languages:         {', '.join(args.languages)}")
    print(f"  Samples:           {args.num_samples if args.num_samples > 0 else 'all'} per language")
    print(f"  Training data:     {args.training_data_folder if args.training_data_folder else 'not provided (no split filtering)'}")
    print(f"  Output dir:        {args.output_dir}")
    print()
    
    # Load training contexts for split filtering
    training_contexts = set()
    if args.training_data_folder:
        training_folder = Path(args.training_data_folder)
        if training_folder.exists():
            training_contexts = load_training_context_ids(training_folder, args.languages)
        else:
            print(f"\nTraining data folder not found: {training_folder}")
            print(f"Will evaluate all data (no split filtering)")
    
    # Load XQuAD samples
    samples = load_xquad_samples(args.languages, num_samples_per_lang=num_samples)
    
    if not samples:
        print("No samples loaded!")
        return
    
    # Filter samples by split and apply random sampling for diversity
    if training_contexts and args.split in ['train', 'dev']:
        filtered_samples = {}
        for lang, lang_samples in samples.items():
            filtered = []
            for sample in lang_samples:
                context_id = sample['context'][:200]
                is_training = context_id in training_contexts
                
                if args.split == "train" and is_training:
                    filtered.append(sample)
                elif args.split == "dev" and not is_training:
                    filtered.append(sample)
            
            if filtered:
                # Randomly sample to ensure diversity across contexts
                if num_samples and len(filtered) > num_samples:
                    import random
                    random.seed(42)  # For reproducibility
                    filtered = random.sample(filtered, num_samples)
                
                filtered_samples[lang] = filtered
        samples = filtered_samples
    
    # Evaluate
    results = evaluate_xquad(
        model_path=args.model_path,
        samples=samples,
        output_dir=Path(args.output_dir),
        max_answer_tokens=args.max_answer_tokens,
        training_contexts=training_contexts
    )
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
