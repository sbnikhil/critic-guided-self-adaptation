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
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
from typing import Optional, Dict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_training_data(input_folder, languages=None, max_contexts_per_language=None):
    import random
    examples = []
    
    input_path = Path(input_folder)
    pattern = "*_best_edits.json"
    
    for file_path in input_path.glob(pattern):
        lang = file_path.stem.replace('_best_edits', '')
        
        if languages and lang not in languages:
            continue
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Sample contexts if limit specified
        if max_contexts_per_language and len(data) > max_contexts_per_language:
            random.seed(42)  # Reproducible sampling
            data = random.sample(data, max_contexts_per_language)
            print(f"  {lang}: sampled {max_contexts_per_language}/{len(data)} contexts")
        
        for article in data:
            context = article['context']
            for edit in article['edits']:
                # Create training example: context → edit
                prompt = f"{context}\n\n"
                examples.append({
                    'text': prompt + edit['generated_text'],
                    'language': lang
                })
    
    print(f"Loaded {len(examples)} training examples")
    return examples


def load_anchor_dataset(anchor_path: str):
    """
    Load pre-computed anchor dataset.
    
    Returns:
        List of dicts with {input_ids, attention_mask, language, h_old}
    """
    print(f"\nLoading anchor dataset from {anchor_path}")
    anchors = torch.load(anchor_path)
    
    # Statistics
    lang_counts = {}
    for anchor in anchors:
        lang = anchor['language']
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
    
    print(f"  Loaded {len(anchors)} anchors")
    print(f"  Representation shape: {anchors[0]['h_old'].shape}")
    print(f"  Languages: {sorted(lang_counts.keys())}")
    for lang in sorted(lang_counts.keys()):
        print(f"    {lang}: {lang_counts[lang]} anchors")
    
    return anchors


class AnchorDataset(torch.utils.data.Dataset):
    """Dataset wrapper for anchor examples."""
    
    def __init__(self, anchors):
        self.anchors = anchors
    
    def __len__(self):
        return len(self.anchors)
    
    def __getitem__(self, idx):
        return self.anchors[idx]


def anchor_collate_fn(batch):
    """Collate function for anchor dataloader."""
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'language': [item['language'] for item in batch],
        'h_old': torch.stack([item['h_old'] for item in batch])
    }


class AnchoringTrainer(Trainer):
    """
    Custom Trainer with representation anchoring support.
    
    Adds anchoring loss: L_total = L_sft + lambda * L_repr
    where L_repr = MSE(h_new, h_old) with optional language weighting.
    """
    
    def __init__(
        self,
        anchor_dataloader: DataLoader,
        anchor_lambda: float = 0.1,
        anchor_lang_weights: Optional[Dict[str, float]] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.anchor_dataloader = anchor_dataloader
        self.anchor_lambda = anchor_lambda
        self.anchor_lang_weights = anchor_lang_weights or {}
        
        # Create infinite iterator for anchors
        self.anchor_iter = iter(self.anchor_dataloader)
        
        print(f"\nAnchoring Configuration:")
        print(f"  Lambda: {self.anchor_lambda}")
        print(f"  Language weights: {self.anchor_lang_weights if self.anchor_lang_weights else 'None (all 1.0)'}")
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute combined loss: L_total = L_sft + lambda * L_repr
        """
        # 1. Standard SFT loss on new task
        outputs = model(**inputs)
        loss_sft = outputs.loss  # usually float32 under mixed precision
        
        # 2. Anchoring loss
        try:
            # Get next anchor batch
            anchor_batch = next(self.anchor_iter)
        except StopIteration:
            # Restart iterator
            self.anchor_iter = iter(self.anchor_dataloader)
            anchor_batch = next(self.anchor_iter)
        
        # Move anchor inputs to device (keep original dtype)
        anchor_input_ids = anchor_batch['input_ids'].to(model.device)
        anchor_attention_mask = anchor_batch['attention_mask'].to(model.device)
        anchor_h_old = anchor_batch['h_old'].to(model.device)  # keep original dtype (likely float32)
        anchor_languages = anchor_batch['language']
        
        # Forward pass on anchors to get current representations
        anchor_outputs = model(
            input_ids=anchor_input_ids,
            attention_mask=anchor_attention_mask,
            output_hidden_states=True
        )
        
        # Extract last hidden state and mean pool
        # Shape: [batch_size, seq_len, hidden_size]
        last_hidden = anchor_outputs.hidden_states[-1]  # [B, T, H], fp16 under autocast
        
        # Mean pool over sequence: [batch_size, seq_len, H] → [batch_size, H]
        # Upcast to float32 for loss computation
        h_new = last_hidden.mean(dim=1).float()  # [B, H], float32
        anchor_h_old = anchor_h_old.float()  # [B, H], float32
        
        # Compute MSE loss with language weighting
        batch_size = h_new.size(0)
        loss_repr_sum = torch.zeros((), device=h_new.device, dtype=torch.float32)
        
        for i in range(batch_size):
            lang = anchor_languages[i]
            weight = float(self.anchor_lang_weights.get(lang, 1.0))
            
            # MSE for this example (scalar float32)
            mse = torch.nn.functional.mse_loss(h_new[i], anchor_h_old[i], reduction='mean')
            loss_repr_sum = loss_repr_sum + weight * mse
        
        loss_repr = loss_repr_sum / batch_size  # float32
        
        # 3. Combine losses - ensure anchoring term matches loss_sft dtype
        loss_repr = loss_repr.to(loss_sft.dtype)
        loss_total = loss_sft + self.anchor_lambda * loss_repr  # both same dtype
        
        # Log losses (will appear in training logs)
        self.log({
            'loss_sft': float(loss_sft.detach().cpu()),
            'loss_repr': float(loss_repr.detach().cpu()),
            'loss_total': float(loss_total.detach().cpu())
        })
        
        return (loss_total, outputs) if return_outputs else loss_total



def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Continual Learning SFT with Representation Anchoring',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Data arguments
    parser.add_argument('--input-folder', type=str, required=True,
                       help='Folder with *_best_edits.json files')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for model checkpoints')
    parser.add_argument('--languages', nargs='+',
                       help='Languages to train on (default: all)')
    parser.add_argument('--max-contexts-per-language', type=int,
                       help='Maximum number of contexts to use per language (default: use all)')
    
    # Model arguments
    parser.add_argument('--base-model', type=str, required=True,
                       help='Base model checkpoint (LoRA adapter from previous phase)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Training batch size per device')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=8,
                       help='Gradient accumulation steps')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--max-length', type=int, default=512,
                       help='Maximum sequence length')
    
    # Anchoring arguments (REQUIRED for continual learning)
    parser.add_argument('--anchor-dataset', type=str, required=True,
                       help='Path to anchor dataset (.pt file from build_anchors.py)')
    parser.add_argument('--anchor-lambda', type=float, default=0.1,
                       help='Weight for anchoring loss (default: 0.1)')
    parser.add_argument('--anchor-batch-size', type=int,
                       help='Anchor batch size (default: same as --batch-size)')
    parser.add_argument('--anchor-lang-weights', type=str,
                       help='JSON string with language weights, e.g. \'{"fi": 2.0, "sw": 2.0}\'')
    
    args = parser.parse_args()
    
    # Parse anchor language weights if provided
    anchor_lang_weights = {}
    if args.anchor_lang_weights:
        anchor_lang_weights = json.loads(args.anchor_lang_weights)
    
    # Set anchor batch size
    anchor_batch_size = args.anchor_batch_size or args.batch_size
    
    print("=" * 70)
    print("CONTINUAL LEARNING SFT WITH REPRESENTATION ANCHORING")
    print("=" * 70)
    print(f"Input folder: {args.input_folder}")
    print(f"Output: {args.output_dir}")
    print(f"Base model: {args.base_model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size per device: {args.batch_size}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps * torch.cuda.device_count()}")
    print(f"Learning rate: {args.learning_rate}")
    print("\nAnchoring Configuration:")
    print(f"  Anchor dataset: {args.anchor_dataset}")
    print(f"  Lambda: {args.anchor_lambda}")
    print(f"  Anchor batch size: {anchor_batch_size}")
    print(f"  Language weights: {anchor_lang_weights if anchor_lang_weights else 'None (all 1.0)'}")
    print("=" * 70 + "\n")
    
    # Load training data
    examples = load_training_data(args.input_folder, args.languages, args.max_contexts_per_language)
    
    if not examples:
        print("No training examples found!")
        return
    
    # Create dataset
    dataset = Dataset.from_list(examples)
    print(f"\nTraining dataset size: {len(dataset)} examples")
    
    # Load base model (existing LoRA checkpoint from previous phase)
    print(f"\nLoading base model: {args.base_model}")
    base_model_path = Path(args.base_model)
    
    if not (base_model_path / "adapter_config.json").exists():
        print(f"Error: {args.base_model} does not appear to be a LoRA checkpoint!")
        print("Expected to find adapter_config.json")
        print("For continual learning, you must provide a LoRA checkpoint from the previous phase.")
        return
    
    # Read adapter config to get base model name
    with open(base_model_path / "adapter_config.json", 'r') as f:
        adapter_config = json.load(f)
    
    base_model_name = adapter_config.get("base_model_name_or_path", "Qwen/Qwen2.5-7B-Instruct")
    print(f"Loading base model: {base_model_name}")
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Load existing LoRA adapter from previous phase
    print(f"Loading LoRA adapter from previous phase: {args.base_model}")
    model = PeftModel.from_pretrained(
        base_model,
        args.base_model,
        is_trainable=True  # Make adapter trainable for continual learning
    )
    print("LoRA checkpoint loaded - will continue training this adapter")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    
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
    
    print("\nTokenizing training dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text', 'language']
    )
    
    # Load anchor dataset (REQUIRED)
    print("\nLoading anchor dataset...")
    anchors = load_anchor_dataset(args.anchor_dataset)
    anchor_dataset = AnchorDataset(anchors)
    anchor_dataloader = DataLoader(
        anchor_dataset,
        batch_size=anchor_batch_size,
        shuffle=True,
        collate_fn=anchor_collate_fn,
        num_workers=0,  # Keep 0 to avoid multiprocessing issues
        pin_memory=True
    )
    print(f"Anchor dataloader created with batch size {anchor_batch_size}")
    
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
    
    # Create AnchoringTrainer
    print("\nInitializing AnchoringTrainer...")
    trainer = AnchoringTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        anchor_dataloader=anchor_dataloader,
        anchor_lambda=args.anchor_lambda,
        anchor_lang_weights=anchor_lang_weights
    )
    
    # Train
    print("\nStarting continual learning training...")
    print("=" * 70)
    trainer.train()
    
    # Save final LoRA model
    final_dir = Path(args.output_dir) / "final"
    print(f"\nSaving LoRA adapter to {final_dir}")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"LoRA adapter saved (for evaluation)")
    
    # Also save merged model for next phase 
    merged_dir = Path(args.output_dir) / "final_merged"
    print(f"\nSaving merged model to {merged_dir}")
    print("Merging LoRA into base weights...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    
    print("\n" + "=" * 70)
    print("CONTINUAL LEARNING TRAINING COMPLETE!")
    print("=" * 70)
    print(f"LoRA adapter saved to:  {final_dir}")
    print(f"Merged model saved to:  {merged_dir}")
    print("\nNext steps:")
    print("1. Evaluate model on both old and new tasks to check for forgetting")
    print("2. If more phases needed, build new anchor dataset:")
    print(f"   python src/build_anchors.py --model-path {final_dir} ...")


if __name__ == '__main__':
    main()
