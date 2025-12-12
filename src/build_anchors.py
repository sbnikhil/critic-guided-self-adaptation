#!/usr/bin/env python3
import json
import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
from typing import List, Dict
import random


def load_model_and_tokenizer(model_path: str):
    """
    Load model (LoRA or merged) and tokenizer.
    
    Args:
        model_path: Path to LoRA checkpoint or merged model
        
    Returns:
        model, tokenizer
    """
    model_path = Path(model_path)
    is_lora = (model_path / "adapter_config.json").exists()
    
    print(f"Loading model from {model_path}")
    
    if is_lora:
        print("  Detected LoRA checkpoint")
        # Load adapter config to get base model
        with open(model_path / "adapter_config.json", 'r') as f:
            adapter_config = json.load(f)
        
        base_model_name = adapter_config.get("base_model_name_or_path", "Qwen/Qwen2.5-7B-Instruct")
        print(f"  Loading base model: {base_model_name}")
        
        # Load base + LoRA
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )
        model = PeftModel.from_pretrained(base_model, str(model_path))
        print("  LoRA model loaded")
    else:
        print("  Loading merged model")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            device_map="auto",
            torch_dtype=torch.float16
        )
        print("  Merged model loaded")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()  # Set to eval mode
    return model, tokenizer


def load_training_data(data_folder: str, languages: List[str], samples_per_lang: int):
    """
    Load and sample training examples from best_edits.json files.
    
    Args:
        data_folder: Folder containing *_best_edits.json files
        languages: List of language codes to include
        samples_per_lang: Number of examples to sample per language
        
    Returns:
        List of dicts with {text, language}
    """
    examples = []
    data_path = Path(data_folder)
    
    print(f"\nLoading training data from {data_folder}")
    print(f"  Languages: {languages}")
    print(f"  Samples per language: {samples_per_lang}")
    
    for lang in languages:
        file_path = data_path / f"{lang}_best_edits.json"
        
        if not file_path.exists():
            print(f"  {lang}: File not found, skipping")
            continue
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Flatten all edits from all articles
        lang_examples = []
        for article in data:
            context = article['context']
            for edit in article['edits']:
                # Same format as training
                prompt = f"{context}\n\n"
                lang_examples.append({
                    'text': prompt + edit['generated_text'],
                    'language': lang
                })
        
        # Sample N examples
        if len(lang_examples) > samples_per_lang:
            sampled = random.sample(lang_examples, samples_per_lang)
        else:
            sampled = lang_examples
        
        examples.extend(sampled)
        print(f"  {lang}: {len(sampled)} examples (from {len(lang_examples)} available)")
    
    print(f"\n  Total: {len(examples)} anchor examples")
    return examples


def extract_representations(model, tokenizer, examples: List[Dict], max_length: int = 512):
    """
    Extract mean-pooled representations from model for each example.
    
    Args:
        model: Model to extract representations from
        tokenizer: Tokenizer
        examples: List of {text, language} dicts
        max_length: Max sequence length
        
    Returns:
        List of dicts with {input_ids, attention_mask, language, h_old}
    """
    anchors = []
    device = next(model.parameters()).device
    
    print(f"\nExtracting representations...")
    print(f"  Device: {device}")
    print(f"  Max length: {max_length}")
    
    with torch.no_grad():
        for example in tqdm(examples, desc="Processing"):
            # Tokenize
            inputs = tokenizer(
                example['text'],
                return_tensors='pt',
                max_length=max_length,
                truncation=True,
                padding='max_length'
            )
            
            # Move to device
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            
            # Forward pass with hidden states
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            # Extract last hidden state: [1, seq_len, hidden_size]
            last_hidden = outputs.hidden_states[-1]  # Last layer
            
            # Mean pool over sequence dimension (dim=1)
            # Shape: [1, seq_len, H] to [1, H]
            h_old = last_hidden.mean(dim=1).squeeze(0)  # [H]
            
            # Store anchor
            anchors.append({
                'input_ids': input_ids.cpu().squeeze(0),  # [seq_len]
                'attention_mask': attention_mask.cpu().squeeze(0),  # [seq_len]
                'language': example['language'],
                'h_old': h_old.cpu()  # [H] - representation vector
            })
    
    print(f"  Extracted {len(anchors)} representations")
    print(f"  Representation shape: {anchors[0]['h_old'].shape}")
    
    return anchors


def save_anchors(anchors: List[Dict], output_path: str):
    """Save anchor dataset to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving anchors to {output_path}")
    torch.save(anchors, output_path)
    print(f"  Saved {len(anchors)} anchors")
    
    # Print statistics
    lang_counts = {}
    total_size = 0
    for anchor in anchors:
        lang = anchor['language']
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
        # Estimate size
        total_size += anchor['input_ids'].numel() * 4  # int32
        total_size += anchor['attention_mask'].numel() * 4
        total_size += anchor['h_old'].numel() * 2  # float16
    
    print(f"\nAnchor Statistics:")
    print(f"  Total anchors: {len(anchors)}")
    print(f"  File size: ~{total_size / (1024**2):.1f} MB")
    print(f"\n  Per language:")
    for lang in sorted(lang_counts.keys()):
        print(f"    {lang}: {lang_counts[lang]} anchors")


def main():
    parser = argparse.ArgumentParser(description='Build anchor dataset for representation anchoring')
    
    # Model arguments
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint (LoRA or merged)')
    
    # Data arguments
    parser.add_argument('--data-folder', type=str, nargs='+', required=True,
                       help='One or more folders containing *_best_edits.json files (space-separated)')
    parser.add_argument('--languages', nargs='+', required=True,
                       help='Languages to include in anchor set')
    parser.add_argument('--samples-per-lang', type=int, default=100,
                       help='Number of examples to sample per language PER FOLDER (default: 100)')
    
    # Output arguments
    parser.add_argument('--output-path', type=str, required=True,
                       help='Output path for anchor dataset (e.g., results/anchors/tydi_M1.pt)')
    
    # Processing arguments
    parser.add_argument('--max-length', type=int, default=512,
                       help='Maximum sequence length (default: 512)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for sampling (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("=" * 70)
    print("BUILD ANCHOR DATASET")
    print("=" * 70)
    print(f"Model: {args.model_path}")
    print(f"Data folder: {args.data_folder}")
    print(f"Languages: {args.languages}")
    print(f"Samples per language: {args.samples_per_lang}")
    print(f"Output: {args.output_path}")
    print(f"Seed: {args.seed}")
    print("=" * 70)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    
    # Load training data from all folders
    all_examples = []
    if isinstance(args.data_folder, list):
        for folder in args.data_folder:
            print(f"\n{'='*70}")
            print(f"Loading from: {folder}")
            print('='*70)
            examples = load_training_data(
                folder,
                args.languages,
                args.samples_per_lang
            )
            all_examples.extend(examples)
    else:
        all_examples = load_training_data(
            args.data_folder,
            args.languages,
            args.samples_per_lang
        )
    
    if not all_examples:
        print("No examples loaded!")
        return
    
    print(f"\n{'='*70}")
    print(f"TOTAL: {len(all_examples)} examples across all folders")
    print('='*70)
    
    # Extract representations
    anchors = extract_representations(
        model,
        tokenizer,
        all_examples,
        max_length=args.max_length
    )
    
    # Save anchors
    save_anchors(anchors, args.output_path)
    
    print("\n" + "=" * 70)
    print("ANCHOR BUILDING COMPLETE!")
    print("=" * 70)
    print(f"\nAnchor file: {args.output_path}")
    print(f"Total anchors: {len(anchors)}")
    print("\nUsage in training:")
    print(f"  --anchor-dataset {args.output_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
