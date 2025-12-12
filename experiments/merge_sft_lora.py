#!/usr/bin/env python3
import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json

def merge_sft_lora(lora_checkpoint_path: str, output_path: str):
    """
    Merge LoRA adapters into base model and save as full model.
    
    Args:
        lora_checkpoint_path: Path to LoRA checkpoint (e.g., results/sft_only/checkpoints/final)
        output_path: Path to save merged model (e.g., results/sft_only/checkpoints/final_merged)
    """
    lora_path = Path(lora_checkpoint_path)
    output_path = Path(output_path)
    
    print("=" * 80)
    print("MERGE SFT LORA INTO BASE MODEL")
    print("=" * 80)
    print(f"Input LoRA checkpoint: {lora_path}")
    print(f"Output merged model:   {output_path}")
    print()
    
    # Read adapter config to get base model name
    adapter_config_path = lora_path / "adapter_config.json"
    if not adapter_config_path.exists():
        print(f"Error: {adapter_config_path} not found!")
        print(f"   {lora_path} doesn't appear to be a LoRA checkpoint.")
        return False
    
    with open(adapter_config_path, 'r') as f:
        adapter_config = json.load(f)
    
    base_model_name = adapter_config.get('base_model_name_or_path', 'Qwen/Qwen2.5-7B-Instruct')
    print(f"Base model: {base_model_name}")
    print()
    
    # Load base model
    print(f"Loading base model from HuggingFace...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    print("   Base model loaded")
    
    # Load LoRA adapters
    print(f"Loading LoRA adapters from {lora_path}...")
    model = PeftModel.from_pretrained(base_model, str(lora_path))
    print("   LoRA adapters loaded")
    
    # Merge
    print("Merging LoRA adapters into base model...")
    merged_model = model.merge_and_unload()
    print("   LoRA merged successfully")
    
    # Save merged model
    print(f"Saving merged model to {output_path}...")
    output_path.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(output_path))
    print("   Model saved")
    
    # Save tokenizer
    print(f"Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(lora_path))
    tokenizer.save_pretrained(str(output_path))
    print("   Tokenizer saved")
    
    # Verify
    print()
    print("=" * 80)
    print("MERGE COMPLETE!")
    print("=" * 80)
    print(f"Merged model saved to: {output_path}")
    print()
    print("This merged model contains:")
    print("  • Base Qwen/Qwen2.5-7B-Instruct weights")
    print("  • SFT improvements from LoRA (baked in)")

    
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Merge SFT LoRA into base model')
    parser.add_argument('--lora-checkpoint', type=str, required=True,
                       help='Path to LoRA checkpoint (e.g., results/sft_only/checkpoints/final)')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Path to save merged model (e.g., results/sft_only/checkpoints/final_merged)')
    
    args = parser.parse_args()
    
    success = merge_sft_lora(args.lora_checkpoint, args.output_dir)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
