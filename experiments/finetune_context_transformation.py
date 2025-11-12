#!/usr/bin/env python3
"""
Context Transformation Fine-Tuning (SEAL Approach)

SEAL's Training Strategy:
1. Train: context → synthetic_data (ALL data, no validation split)
2. Test: question → answer (WITHOUT context - different task!)

This script does step 1: Fine-tune on context transformations.
Use evaluate_no_context.py for step 2: Test on QA without context.

Goal: Internalize knowledge from passages, not just memorize Q&A pairs.

Usage:
    python finetune_context_transformation.py --epochs 3 --lr 2e-5
"""


import os
import sys
import json
import random
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import torch
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

# Add src to path
sys.path.insert(0, 'src')
from metrics import MetricsCalculator
from constants import LANGUAGE_NAMES


# Configuration
MODEL_NAME = "Qwen/Qwen2.5-7B"
CHECKPOINT_DIR = Path("results/checkpoints/context_transformation")
METRICS_DIR = Path("results/metrics")
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
MAX_LENGTH = 512

# Ensure directories exist
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

print("="*80)
print("CONTEXT TRANSFORMATION FINE-TUNING")
print("="*80)
print("Goal: Train model to transform contexts across formats and languages")
print("Method: Format-specific instruction → Transformed context")

# ========== Step 1: Load Best Edits ==========
print("\n" + "="*80)
print("STEP 1: Loading Best Edits")
print("="*80)


def load_best_edits(best_edits_dir):
    """Load all best edits grouped by language"""
    edits_by_lang = defaultdict(list)
    best_edits_dir = Path(best_edits_dir)
    if not best_edits_dir.exists():
        print(f"ERROR: Input folder {best_edits_dir} does not exist.")
        sys.exit(1)
    for file in best_edits_dir.glob("*_best_edits.json"):
        lang = file.stem.replace("_best_edits", "")
        print(f"Loading {file.name}...")
        with open(file, "r", encoding="utf-8") as f:
            edits = json.load(f)
            edits_by_lang[lang].extend(edits)
    total = sum(len(v) for v in edits_by_lang.values())
    print(f"Loaded {total} best edits across {len(edits_by_lang)} languages")
    return edits_by_lang

# ========== Step 2: Prepare Format-Specific Training Data ==========
print("\n" + "="*80)
print("STEP 2: Preparing Format-Specific Training Data")
print("="*80)

def create_format_instruction(format_type: str, lang_name: str) -> str:
    """Create format-specific instruction prompts"""
    instructions = {
        "rewrite": f"Rewrite the following passage in {lang_name} using different words while keeping the exact same meaning:",
        "implications": f"List factual implications from the following passage in {lang_name}:",
        "chain_of_thought": f"Analyze the following passage step-by-step in {lang_name}:",
        "self_qa": f"Create question-answer pairs from the following passage in {lang_name}:"
    }
    return instructions.get(format_type, f"Transform the following passage in {lang_name}:")

def prepare_training_data_by_format(edits_by_lang, min_quality_score=6.0):
    """
    Prepare training examples for CONTEXT TRANSFORMATION (not QA!)
    
    Training pairs: (format_instruction + context) → transformed_context
    """
    examples_by_format = defaultdict(list)
    examples_by_lang = defaultdict(list)
    
    stats = {
        'total': 0,
        'filtered_low_quality': 0,
        'filtered_invalid': 0,
        'used': 0,
        'by_format': defaultdict(int)
    }
    
    for lang, edits in edits_by_lang.items():
        lang_name = LANGUAGE_NAMES.get(lang, lang)
        
        for edit in edits:
            stats['total'] += 1
            
            best_edit = edit.get('best_edit', {})
            if not best_edit:
                continue
            
            # Filter by validation
            if not best_edit.get('is_valid', True):
                stats['filtered_invalid'] += 1
                continue
            
            # Filter by critic score
            critic_score = edit.get('critic_score', 0.0)
            if critic_score < min_quality_score:
                stats['filtered_low_quality'] += 1
                continue
            
            context = edit['context']
            generated_text = best_edit.get('generated_text', '')
            format_type = best_edit.get('format_type', 'rewrite')
            
            if not generated_text or len(generated_text) < 20:
                continue
            
            # Create format-specific training pair
            instruction = create_format_instruction(format_type, lang_name)
            
            training_text = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{instruction}

Passage: {context}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{generated_text}<|eot_id|>"""
            
            example = {
                'text': training_text,
                'language': lang,
                'format': format_type,
                'context_length': len(context.split()),
                'output_length': len(generated_text.split()),
                'semantic_similarity': best_edit.get('semantic_similarity', 0.0),
                'critic_score': critic_score
            }
            
            examples_by_format[format_type].append(example)
            examples_by_lang[lang].append(example)
            stats['by_format'][format_type] += 1
            stats['used'] += 1
    
    print(f"\nDataset Statistics:")
    print(f"  Total edits: {stats['total']}")
    print(f"  Filtered (invalid): {stats['filtered_invalid']}")
    print(f"  Filtered (low quality < {min_quality_score}): {stats['filtered_low_quality']}")
    print(f"  Used for training: {stats['used']}")
    print(f"\nBy format:")
    for fmt, count in stats['by_format'].items():
        print(f"  {fmt}: {count}")
    print(f"\nBy language:")
    for lang, examples in examples_by_lang.items():
        print(f"  {lang}: {len(examples)}")
    
    return examples_by_format, examples_by_lang, stats

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=7, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--samples', type=int, default=None, help='Max samples to use (for testing)')
    parser.add_argument('--input-folder', type=str, required=True, help='Path to folder containing *_best_edits.json files')
    return parser.parse_args()

args = parse_args()

edits_by_lang = None

def main():
    global edits_by_lang
    edits_by_lang = load_best_edits(args.input_folder)
    # ========== Step 2: Prepare Format-Specific Training Data ==========
    print("\n" + "="*80)
    print("STEP 2: Preparing Format-Specific Training Data")
    print("="*80)
    examples_by_format, examples_by_lang, data_stats = prepare_training_data_by_format(edits_by_lang)

    # Combine all examples for training (SEAL approach: use ALL data)
    all_examples = []
    for format_examples in examples_by_format.values():
        all_examples.extend(format_examples)

    # Shuffle for randomness
    random.shuffle(all_examples)

    print(f"\nTotal training examples: {len(all_examples)}")
    print("(Following SEAL: using ALL best edits for training, no held-out validation)")
    print("(Evaluation will be on separate QA task WITHOUT context)")

    train_examples = all_examples
    val_examples = []  # No validation split - SEAL doesn't use one

    # ========== Step 3: Initialize Model ==========
    print("\n" + "="*80)
    print("STEP 3: Initializing Model")
    print("="*80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )

    # Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    print("Applying LoRA...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ========== Step 4: Create Dataset ==========
    print("\n" + "="*80)
    print("STEP 4: Creating Dataset")
    print("="*80)

    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=MAX_LENGTH,
            padding='max_length'
        )

    train_dataset = Dataset.from_list(train_examples)
    train_dataset = train_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=['text', 'language', 'format', 'context_length', 'output_length', 'semantic_similarity', 'critic_score']
    )

    # No validation dataset - SEAL doesn't use one
    # Real test is on QA task (evaluate_no_context.py)
    val_dataset = None

    # ========== Step 5: Fine-Tune ==========
    print("\n" + "="*80)
    print("STEP 5: Fine-Tuning on Context Transformations")
    print("="*80)

    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=args.lr,
        warmup_steps=100,
        logging_steps=10,
        eval_strategy="no",  # No validation dataset (SEAL approach)
        save_strategy="epoch",
        load_best_model_at_end=False,  # No validation to compare
        save_total_limit=3,
        fp16=device == "cuda",
        report_to="none"
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,  # No validation dataset (SEAL approach)
        data_collator=data_collator
    )

    print(f"Starting training for {args.epochs} epochs with LR={args.lr}...")
    trainer.train()

    # Save final model
    final_model_path = CHECKPOINT_DIR / "final"
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))
    print(f"\nModel saved to {final_model_path}")

    # ========== Step 6: Evaluation ==========
    print("\n" + "="*80)
    print("STEP 6: Evaluation - Format Transformation Capability")
    print("="*80)

    print("\nEvaluating model's ability to generate format transformations...")
    # ... (rest of evaluation code can go here, using examples_by_format, etc.)


if __name__ == "__main__":
    main()

training_args = TrainingArguments(
    output_dir=str(CHECKPOINT_DIR),
    num_train_epochs=args.epochs,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=args.lr,
    warmup_steps=100,
    logging_steps=10,
    eval_strategy="no",  # No validation dataset (SEAL approach)
    save_strategy="epoch",
    load_best_model_at_end=False,  # No validation to compare
    save_total_limit=3,
    fp16=device == "cuda",
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=None,  # No validation dataset (SEAL approach)
    data_collator=data_collator
)

print(f"Starting training for {args.epochs} epochs with LR={args.lr}...")
trainer.train()

# Save final model
final_model_path = CHECKPOINT_DIR / "final"
trainer.save_model(str(final_model_path))
tokenizer.save_pretrained(str(final_model_path))
print(f"\nModel saved to {final_model_path}")

# ========== Step 6: Evaluation ==========
print("\n" + "="*80)
print("STEP 6: Evaluation - Format Transformation Capability")
print("="*80)

print("\nEvaluating model's ability to generate format transformations...")

def evaluate_transformation_capability(model, tokenizer, test_examples, num_samples=30):
    """
    Evaluate if model can generate format transformations
    NOT evaluating QA accuracy - evaluating transformation quality!
    """
    results_by_format = defaultdict(list)
    
    # Sample from each format
    for format_type, format_examples in examples_by_format.items():
        samples = random.sample(format_examples, min(num_samples, len(format_examples)))
        
        print(f"\nTesting {format_type} format...")
        
        for sample in tqdm(samples[:10], desc=f"Generating {format_type}"):  # Limit to 10 for speed
            # Extract context from training text
            text = sample['text']
            # Find the passage content
            import re
            passage_match = re.search(r'Passage: (.+?)<\|eot_id\|>', text, re.DOTALL)
            if not passage_match:
                continue
            
            context = passage_match.group(1).strip()
            lang = sample['language']
            lang_name = LANGUAGE_NAMES.get(lang, lang)
            
            # Create prompt for generation
            instruction = create_format_instruction(format_type, lang_name)
            prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{instruction}

Passage: {context}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.5,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
            
            # Measure semantic similarity
            from sentence_transformers import SentenceTransformer
            sem_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            
            emb1 = sem_model.encode([context])[0]
            emb2 = sem_model.encode([generated])[0]
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            
            results_by_format[format_type].append({
                'context': context,
                'generated': generated,
                'semantic_similarity': float(similarity),
                'language': lang
            })
    
    # Calculate statistics
    print("\n" + "="*80)
    print("TRANSFORMATION QUALITY RESULTS")
    print("="*80)
    print(f"\n{'Format':<20} {'Avg SemSim':<12} {'Samples':<10}")
    print("-" * 50)
    
    all_sims = []
    for format_type, results in results_by_format.items():
        sims = [r['semantic_similarity'] for r in results]
        avg_sim = np.mean(sims)
        all_sims.extend(sims)
        print(f"{format_type:<20} {avg_sim:<12.4f} {len(results):<10}")
    
    print("-" * 50)
    print(f"{'Overall':<20} {np.mean(all_sims):<12.4f} {len(all_sims):<10}")
    
    # Save results
    eval_results = {
        'by_format': {
            fmt: {
                'semantic_similarity': float(np.mean([r['semantic_similarity'] for r in results])),
                'num_samples': len(results)
            }
            for fmt, results in results_by_format.items()
        },
        'overall': {
            'semantic_similarity': float(np.mean(all_sims)),
            'num_samples': len(all_sims)
        },
        'training_config': {
            'epochs': args.epochs,
            'learning_rate': args.lr,
            'total_training_samples': len(train_examples),
            'data_quality_filter': 'critic_score >= 6.0'
        },
        'timestamp': datetime.now().isoformat()
    }
    
    results_file = METRICS_DIR / "context_transformation_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {results_file}")
    
    return eval_results

# Run evaluation
eval_results = evaluate_transformation_capability(model, tokenizer, val_examples)

print("\n" + "="*80)
print("COMPLETED!")
print("="*80)
print("\nNext steps:")
print("1. Check semantic similarity scores (target: >= 0.7)")
print("2. If good, test on downstream QA task")
print("3. Compare to baseline (no transformations)")
