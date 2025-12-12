#!/usr/bin/env python3


import json
import sys
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_training_data(input_folder, languages=None):
    """Load best_edits.json files and create training examples."""
    examples = []
    
    input_path = Path(input_folder)
    pattern = "*_best_edits.json"
    
    for file_path in input_path.glob(pattern):
        lang = file_path.stem.replace('_best_edits', '')
        
        if languages and lang not in languages:
            continue
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for article in data:
            context = article['context']
            for edit in article['edits']:
                # Create training example: context → edit
                # Generic prompt that works for both QA and reasoning tasks
                prompt = f"{context}\n\n"
                examples.append({
                    'text': prompt + edit['generated_text'],
                    'language': lang
                })
    
    print(f"Loaded {len(examples)} training examples")
    return examples


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='SFT training on critic-approved edits')
    parser.add_argument('--input-folder', type=str, required=True,
                       help='Folder with *_best_edits.json files')
    parser.add_argument('--output-dir', type=str, default='results/sft_only/checkpoints',
                       help='Output directory for model')
    parser.add_argument('--base-model', type=str, default='Qwen/Qwen2.5-7B-Instruct',
                       help='Base model to fine-tune')
    parser.add_argument('--languages', nargs='+',
                       help='Languages to train on (default: all)')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Training batch size per device')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=8,
                       help='Gradient accumulation steps (effective batch = batch_size * accum_steps * num_gpus)')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--max-length', type=int, default=512,
                       help='Maximum sequence length')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("SFT TRAINING")
    print("=" * 70)
    print(f"Input folder: {args.input_folder}")
    print(f"Output: {args.output_dir}")
    print(f"Base model: {args.base_model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size per device: {args.batch_size}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps * torch.cuda.device_count()}")
    print(f"Learning rate: {args.learning_rate}")
    print("=" * 70 + "\n")
    
    # Load data
    examples = load_training_data(args.input_folder, args.languages)
    
    if not examples:
        print("No training examples found!")
        return
    
    # Create dataset
    dataset = Dataset.from_list(examples)
    
    print(f"\nDataset size: {len(dataset)} examples")
    
    # Load model and tokenizer
    print(f"\nLoading model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Apply LoRA
    print("\nApplying LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    
    model = get_peft_model(model, lora_config)
    
    # Enable gradient checkpointing to save memory
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    
    print("\nTrainable parameters:")
    model.print_trainable_parameters()
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=args.max_length,
            padding='max_length'
        )
    
    print("\nTokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text', 'language'])
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    # Train
    print("\nStarting SFT training...")
    print("=" * 70)
    trainer.train()
    
    # Save final model
    final_dir = Path(args.output_dir) / "final"
    print(f"\nSaving final model to {final_dir}")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    print("\n" + "=" * 70)
    print("SFT TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Model saved to: {final_dir}")


if __name__ == '__main__':
    main()
