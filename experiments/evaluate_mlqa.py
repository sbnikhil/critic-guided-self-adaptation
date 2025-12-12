import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Set
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

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
)

# MLQA resource groups (based on language resources and NLP infrastructure)
MLQA_RESOURCE_GROUPS = {
    'HIGH': ['en'],   # High-resource languages
    'MID': ['de'],    # Mid-resource languages  
    'LOW': ['vi']     # Low-resource languages
}

def get_mlqa_path() -> Path:
    """Get path to MLQA data directory"""
    current_dir = Path(__file__).parent
    
    possible_paths = [
        current_dir.parent / "data" / "MLQA_V1",
        current_dir.parent / "data" / "mlqa",
        Path("/afs/cs.wisc.edu/u/n/i/nikhilsb/critic/critic/data/MLQA_V1"),
        Path("/Users/Patron/Desktop/Projects/NLP/critic/data/MLQA_V1"),
        Path("/content/drive/Shareddrives/CS769-NLP/critic/data/MLQA_V1"),
    ]
    
    for data_path in possible_paths:
        if data_path.exists():
            return data_path
    
    raise FileNotFoundError(
        f"MLQA data not found. Tried:\n" + "\n".join(str(p) for p in possible_paths) +
        "\n\nPlease download from: https://github.com/facebookresearch/MLQA"
    )

def load_mlqa_samples(languages: List[str], num_samples_per_lang: int = 100, split: str = "dev") -> Dict:
    """
    Load MLQA samples for specified languages.
    
    MLQA format: monolingual pairs (context and question in same language)
    File structure: dev/dev-context-{lang}-question-{lang}.json
    
    Args:
        languages: List of language codes (e.g., ['en', 'ar', 'de'])
        num_samples_per_lang: Number of samples per language
        split: 'dev' or 'test'
    
    Returns:
        Dict mapping language -> list of QA samples
    """
    mlqa_path = get_mlqa_path()
    data_dir = mlqa_path / split
    print(f"Loading MLQA samples from {data_dir}...")
    
    # MLQA languages: en, ar, de, es, hi, vi, zh
    available_langs = ['en', 'ar', 'de', 'es', 'hi', 'vi', 'zh']
    
    samples = {}
    
    for lang in languages:
        if lang not in available_langs:
            print(f"  Warning: {lang} not available in MLQA, skipping")
            continue
        
        # Monolingual file: both context and question in same language
        file_path = data_dir / f"{split}-context-{lang}-question-{lang}.json"
        
        if not file_path.exists():
            print(f"  Warning: {lang} file not found: {file_path}")
            continue
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        lang_samples = []
        
        # Parse MLQA format (SQuAD-like structure)
        for article in data.get('data', []):
            for paragraph in article.get('paragraphs', []):
                context = paragraph.get('context', '')
                
                for qa in paragraph.get('qas', []):
                    question = qa.get('question', '')
                    answers = qa.get('answers', [])
                    
                    if not answers:
                        continue
                    
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
        
        # Sample for diversity
        if num_samples_per_lang and len(lang_samples) > num_samples_per_lang:
            import random
            random.seed(42)
            lang_samples = random.sample(lang_samples, num_samples_per_lang)
        
        samples[lang] = lang_samples
        print(f"  {lang}: {len(lang_samples)} samples")
    
    return samples

def load_training_context_ids(training_folder: Path, languages: List[str]) -> Set[str]:
    """Load context IDs from training data (best_edits)"""
    training_contexts = set()
    
    for lang in languages:
        best_edits_file = training_folder / f"{lang}_best_edits.json"
        if not best_edits_file.exists():
            continue
        
        with open(best_edits_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for article in data:
            context = article.get('context', '')
            context_id = context[:200]
            training_contexts.add(context_id)
        
        print(f"  {lang}: {len([a for a in data])} training contexts")
    
    return training_contexts

def load_model_and_tokenizer(model_path: str):
    """Load model and tokenizer (handles both base models and LoRA checkpoints)"""
    print(f"\nLoading model from: {model_path}")
    
    num_gpus = torch.cuda.device_count()
    print(f"  Available GPUs: {num_gpus}")
    
    model_path_obj = Path(model_path)
    adapter_config_path = model_path_obj / "adapter_config.json"
    
    max_memory = {i: "10GB" for i in range(num_gpus)}
    max_memory["cpu"] = "30GB"
    
    if adapter_config_path.exists():
        print("  Detected LoRA checkpoint")
        
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config.get('base_model_name_or_path', 'Qwen/Qwen2.5-7B-Instruct')
        
        print(f"  Loading base model: {base_model_name}")
        is_local_base = Path(base_model_name).exists()
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            max_memory=max_memory,
            torch_dtype=torch.float16,
            local_files_only=is_local_base,
            trust_remote_code=True
        )
        
        print(f"  Loading LoRA adapters from: {model_path}")
        model = PeftModel.from_pretrained(model, model_path, local_files_only=True)
        model = model.merge_and_unload()
        
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            local_files_only=is_local_base,
            trust_remote_code=True
        )
        print("  LoRA model loaded and merged")
    else:
        print("  Loading base model")
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
        print("  Base model loaded")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    print(f"  Model memory footprint:")
    for i in range(num_gpus):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"    GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    return model, tokenizer

def generate_answer(model, tokenizer, context: str, question: str, max_tokens: int = 50) -> str:
    """Generate answer for a given question and context"""
    prompt = f"""Answer the question based on the context.

Context: {context}

Question: {question}

Answer:"""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    answer = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
    return answer

def evaluate_mlqa(
    model_path: str,
    samples: Dict[str, List],
    output_dir: Path,
    max_answer_tokens: int = 50,
    training_contexts: Set[str] = None
):
    """Evaluate model on MLQA samples with comprehensive metrics"""
    
    model, tokenizer = load_model_and_tokenizer(model_path)
    metrics_calc = MetricsCalculator()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Results storage
    all_results = []
    per_lang_results = {}
    
    # Process each language
    total_samples = sum(len(v) for v in samples.values())
    print(f"\nEvaluating {total_samples} samples across {len(samples)} languages\n")
    
    for lang, lang_samples in samples.items():
        print(f"\nProcessing {lang.upper()} ({len(lang_samples)} samples)")
        
        lang_results = []
        
        # Generate answers for all samples
        print(f"  Generating answers...")
        predictions = []
        for sample in tqdm(lang_samples, desc=f"  {lang}"):
            predicted_answer = generate_answer(
                model, tokenizer, sample['context'], sample['question'], max_answer_tokens
            )
            predictions.append(predicted_answer)
        
        # Compute traditional metrics (per-sample)
        print(f"  Computing traditional metrics...")
        for sample, predicted_answer in zip(lang_samples, predictions):
            sample['predicted'] = predicted_answer
            sample['span_f1'] = compute_span_f1(predicted_answer, sample['answer'])
            sample['rouge_l'] = compute_rouge_l(predicted_answer, sample['answer'])
            sample['exact_match'] = metrics_calc.exact_match(predicted_answer, sample['answer'])
            sample['is_training_context'] = sample['context'][:200] in training_contexts if training_contexts else False
        
        # Compute BLEURT (batch)
        print(f"  Computing BLEURT scores...")
        bleurt_scores = compute_bleurt(
            predictions=[s['predicted'] for s in lang_samples],
            ground_truths=[s['answer'] for s in lang_samples]
        )
        for sample, bleurt_score in zip(lang_samples, bleurt_scores):
            sample['bleurt'] = float(bleurt_score)
        
        # Compute LLM-as-judge scores (batch)
        print(f"  Computing LLM-as-judge correctness scores...")
        correctness_scores = llm_judge_correctness_batch(
            questions=[s['question'] for s in lang_samples],
            contexts=[s['context'] for s in lang_samples],
            ground_truths=[s['answer'] for s in lang_samples],
            predictions=[s['predicted'] for s in lang_samples],
            languages=[lang] * len(lang_samples)
        )
        
        print(f"  Computing LLM-as-judge quality scores...")
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
        
        print(f"\n  Results for {lang}:")
        print(f"     Exact Match:       {per_lang_results[lang]['exact_match']:.2%}")
        print(f"     Span F1:           {per_lang_results[lang]['span_f1']:.2%}")
        print(f"     ROUGE-L:           {per_lang_results[lang]['rouge_l']:.2%}")
        print(f"     BLEURT:            {per_lang_results[lang]['bleurt']:.4f}")
        print(f"     LLM Correctness:   {per_lang_results[lang]['llm_correctness']:.2f}/5")
        print(f"     LLM Quality:       {per_lang_results[lang]['llm_quality']:.2f}/5")
    
    # Overall aggregation with comprehensive cross-lingual metrics
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
    group_gaps = compute_group_gap(lang_f1_scores, MLQA_RESOURCE_GROUPS)
    
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
    print("\nOVERALL RESULTS")
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
            langs = MLQA_RESOURCE_GROUPS[group]
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
    parser = argparse.ArgumentParser(description="Evaluate model on MLQA dataset")
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
        default=['en','de', 'vi'],
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
        choices=['dev', 'test', 'train'],
        help="Dataset split: 'train' = training contexts, 'dev' = rest (default: dev)"
    )
    parser.add_argument(
        "--training_data_folder",
        type=str,
        default=None,
        help="Folder with *_best_edits.json files (for train/dev split filtering)"
    )
    
    args = parser.parse_args()
    
    num_samples = args.num_samples if args.num_samples > 0 else None
    
    print("MLQA QA EVALUATION")
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
            print(f"\nLoading training contexts from: {training_folder}")
            training_contexts = load_training_context_ids(training_folder, args.languages)
        else:
            print(f"\nWarning: Training data folder not found: {training_folder}")
            print(f"  Will evaluate all data (no split filtering)")
    
    # Load MLQA samples
    samples = load_mlqa_samples(args.languages, num_samples_per_lang=num_samples, split=args.split)
    
    if not samples:
        print("Error: No samples loaded")
        return
    
    # Filter samples by split
    if training_contexts and args.split in ['train', 'dev']:
        print(f"\nFiltering samples for {args.split} split")
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
                if num_samples and len(filtered) > num_samples:
                    import random
                    random.seed(42)
                    filtered = random.sample(filtered, num_samples)
                
                filtered_samples[lang] = filtered
                print(f"  {lang}: {len(filtered)}/{len(lang_samples)} samples")
        
        samples = filtered_samples
    
    # Evaluate
    results = evaluate_mlqa(
        model_path=args.model_path,
        samples=samples,
        output_dir=Path(args.output_dir),
        max_answer_tokens=args.max_answer_tokens,
        training_contexts=training_contexts
    )
    
    print("\nEvaluation complete")

if __name__ == "__main__":
    main()