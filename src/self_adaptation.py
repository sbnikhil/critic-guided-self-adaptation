#!/usr/bin/env python3
"""
Self-Adaptation Pipeline for Multilingual Continual Learning

End-to-end pipeline for:
1. Loading TyDiQA data
2. Generating self-edits (multi-format or QA-only)
3. Selecting best edits with Gemini critic
4. Fine-tuning with LoRA (multi-task across all languages)
5. Comprehensive evaluation with cross-lingual transfer metrics

Usage:
    python self_adaptation.py --format all_formats --samples 10
    python self_adaptation.py --format qa_only --samples 10
    python self_adaptation.py --format no_edits --samples 10
"""

import os
import sys
import json
import argparse
import random
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Any
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import project modules
from data_loader import load_tydiqa_by_language, get_available_languages
from multi_format_self_edit import MultiFormatSelfEditGenerator
from critic import Critic
from metrics import MetricsCalculator
from constants import LANGUAGE_NAMES

# Load environment variables
load_dotenv()


# ========== Configuration ==========

class Config:
    """Pipeline configuration"""
    
    # Model settings
    MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
    
    # LoRA settings
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    # Training settings
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 4
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 3
    WARMUP_STEPS = 100
    MAX_LENGTH = 512
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 3
    EARLY_STOPPING_THRESHOLD = 0.001
    
    # Data settings
    TRAIN_SAMPLES_PER_LANGUAGE = 10
    VAL_SAMPLES_PER_LANGUAGE = 5
    
    # Paths
    RESULTS_DIR = Path("results")
    CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
    METRICS_DIR = RESULTS_DIR / "metrics"
    LOGS_DIR = RESULTS_DIR / "logs"
    
    # Random seed
    RANDOM_SEED = 42
    
    # Critic settings
    CRITIC_BATCH_SIZE = 2
    CRITIC_DELAY_SECONDS = 6.5


# ========== Helper Functions ==========

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_directories():
    """Create necessary directories."""
    Config.RESULTS_DIR.mkdir(exist_ok=True)
    Config.CHECKPOINTS_DIR.mkdir(exist_ok=True)
    Config.METRICS_DIR.mkdir(exist_ok=True)
    Config.LOGS_DIR.mkdir(exist_ok=True)


def save_json(data: Any, filepath: Path):
    """Save data to JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: Path) -> Any:
    """Load data from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


# ========== Data Loading & Preparation ==========

class DataManager:
    """Manage data loading and preparation"""
    
    def __init__(self, format_type: str, samples_per_lang: int):
        """
        Initialize data manager.
        
        Args:
            format_type: 'no_edits', 'qa_only', or 'all_formats'
            samples_per_lang: Number of samples per language
        """
        self.format_type = format_type
        self.samples_per_lang = samples_per_lang
        self.languages = get_available_languages()
        
        # Create format-specific directories
        self.edits_dir = Config.RESULTS_DIR / format_type
        self.best_edits_dir = Config.RESULTS_DIR / "best_edits" / format_type
        self.edits_dir.mkdir(parents=True, exist_ok=True)
        self.best_edits_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data_for_all_languages(
        self,
        split: str = "train"
    ) -> Dict[str, List[Dict]]:
        """
        Load TyDiQA data for all languages.
        
        Args:
            split: 'train' or 'dev'
            
        Returns:
            Dict mapping language to list of articles
        """
        print(f"\n{'='*80}")
        print(f"Loading {split} data for all languages...")
        print(f"{'='*80}")
        
        data_by_language = {}
        
        for lang in self.languages:
            print(f"\nLoading {lang.upper()}...")
            articles = load_tydiqa_by_language(
                lang,
                max_samples=self.samples_per_lang,
                split=split
            )
            data_by_language[lang] = articles
            print(f"  Loaded {len(articles)} articles")
        
        return data_by_language
    
    def prepare_training_validation_split(
        self,
        data_by_language: Dict[str, List[Dict]]
    ) -> Tuple[Dict[str, List[Dict]], Dict[str, List[Dict]]]:
        """
        Split data into training and validation sets.
        
        Args:
            data_by_language: Dict mapping language to articles
            
        Returns:
            Tuple of (train_data, val_data)
        """
        train_data = {}
        val_data = {}
        
        for lang, articles in data_by_language.items():
            # Use 70% for training, 30% for validation
            # Ensure at least 1 sample for training if we have any data
            split_idx = max(1, int(len(articles) * 0.7))
            
            # If we only have 1 sample, use it for both train and val
            if len(articles) == 1:
                train_data[lang] = articles
                val_data[lang] = articles
            else:
                train_data[lang] = articles[:split_idx]
                val_data[lang] = articles[split_idx:]
        
        return train_data, val_data


# ========== Self-Edit Generation ==========

class EditGenerator:
    """Generate self-edits based on format type"""
    
    def __init__(self, format_type: str):
        """
        Initialize edit generator.
        
        Args:
            format_type: 'no_edits', 'qa_only', or 'all_formats'
        """
        self.format_type = format_type
        
        if format_type != 'no_edits':
            self.generator = MultiFormatSelfEditGenerator()
    
    def generate_edits_for_language(
        self,
        language: str,
        articles: List[Dict],
        save_dir: Path
    ) -> List[Dict]:
        """
        Generate edits for a specific language.
        
        Args:
            language: Language code
            articles: List of articles
            save_dir: Directory to save edits
            
        Returns:
            List of QA groups with edits
        """
        print(f"\n{'='*80}")
        print(f"Generating edits for {language.upper()} ({self.format_type})")
        print(f"{'='*80}")
        
        if self.format_type == 'no_edits':
            # Just wrap original QA pairs
            qa_groups = []
            for i, article in enumerate(articles):
                if not article['qa_pairs']:
                    continue
                
                qa = article['qa_pairs'][0]
                qa_groups.append({
                    'article_id': i,
                    'context': article['context'],
                    'original_question': qa['question'],
                    'original_answer': qa['answer'],
                    'language': language,
                    'edits': []  # No edits for baseline
                })
            
            # Save
            output_file = save_dir / f"{language}_edits.json"
            save_json(qa_groups, output_file)
            print(f"Saved {len(qa_groups)} original QA pairs to {output_file}")
            
            return qa_groups
        
        # Generate edits
        qa_groups = []
        
        for i, article in enumerate(tqdm(articles, desc=f"Processing {language}")):
            if not article['qa_pairs']:
                continue
            
            qa = article['qa_pairs'][0]
            edits = []
            
            if self.format_type == 'qa_only':
                # Generate 5 QA-format edits
                for _ in range(5):
                    result = self.generator.generate_edit(
                        context=article['context'],
                        question=qa['question'],
                        answer=qa['answer'],
                        language=language,
                        format_type='self_qa'
                    )
                    edits.append(result)
            
            elif self.format_type == 'all_formats':
                # Generate all 4 formats
                formats = list(self.generator.GENERATION_FORMATS.keys())
                for fmt in formats:
                    result = self.generator.generate_edit(
                        context=article['context'],
                        question=qa['question'],
                        answer=qa['answer'],
                        language=language,
                        format_type=fmt
                    )
                    edits.append(result)
            
            qa_groups.append({
                'article_id': i,
                'context': article['context'],
                'original_question': qa['question'],
                'original_answer': qa['answer'],
                'language': language,
                'edits': edits
            })
        
        # Save
        output_file = save_dir / f"{language}_edits.json"
        save_json(qa_groups, output_file)
        print(f"Saved {len(qa_groups)} QA groups with edits to {output_file}")
        
        return qa_groups
    
    def generate_all_edits(
        self,
        data_by_language: Dict[str, List[Dict]],
        save_dir: Path
    ) -> Dict[str, List[Dict]]:
        """
        Generate edits for all languages.
        
        Args:
            data_by_language: Dict mapping language to articles
            save_dir: Directory to save edits
            
        Returns:
            Dict mapping language to QA groups with edits
        """
        all_edits = {}
        
        for lang, articles in data_by_language.items():
            qa_groups = self.generate_edits_for_language(lang, articles, save_dir)
            all_edits[lang] = qa_groups
        
        # Save summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'format_type': self.format_type,
            'languages': len(all_edits),
            'total_qa_pairs': sum(len(groups) for groups in all_edits.values())
        }
        save_json(summary, save_dir / "summary.json")
        
        return all_edits


# ========== Critic Selection ==========

class CriticSelector:
    """Select best edits using Gemini critic"""
    
    def __init__(self, format_type: str):
        """
        Initialize critic selector.
        
        Args:
            format_type: 'no_edits', 'qa_only', or 'all_formats'
        """
        self.format_type = format_type
        
        if format_type != 'no_edits':
            api_key = os.environ.get('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment")
            self.critic = Critic(api_key=api_key)
    
    def select_best_edits_for_language(
        self,
        language: str,
        qa_groups: List[Dict],
        save_dir: Path
    ) -> List[Dict]:
        """
        Select best edits for a language using critic.
        
        Args:
            language: Language code
            qa_groups: List of QA groups with edits
            save_dir: Directory to save best edits
            
        Returns:
            List of best edits
        """
        print(f"\n{'='*80}")
        print(f"Selecting best edits for {language.upper()}")
        print(f"{'='*80}")
        
        if self.format_type == 'no_edits':
            # No selection needed for baseline
            best_edits = []
            for qa_group in qa_groups:
                best_edits.append({
                    'article_id': qa_group['article_id'],
                    'language': language,
                    'context': qa_group['context'],
                    'original_question': qa_group['original_question'],
                    'original_answer': qa_group['original_answer'],
                    'best_edit': None,  # No edits
                    'critic_approved': True,
                    'critic_score': 10.0
                })
            
            output_file = save_dir / f"{language}_best_edits.json"
            save_json(best_edits, output_file)
            return best_edits
        
        # Use critic to select best edits
        best_edits = []
        batch_size = Config.CRITIC_BATCH_SIZE
        
        for batch_start in range(0, len(qa_groups), batch_size):
            batch_end = min(batch_start + batch_size, len(qa_groups))
            batch_qa = qa_groups[batch_start:batch_end]
            
            if batch_start > 0:
                time.sleep(Config.CRITIC_DELAY_SECONDS)
            
            print(f"  Batch {batch_start//batch_size + 1}: QA {batch_start+1}-{batch_end}")
            batch_results = self.critic.evaluate_language_batch(batch_qa)
            
            for qa_group, result in zip(batch_qa, batch_results):
                selected_idx = int(result.get('selected_index', -1))
                score = float(result.get('score', 0.0))
                reason = result.get('reason', '')
                
                edits = qa_group.get('edits', [])
                
                if 1 <= selected_idx <= len(edits):
                    best_edit = edits[selected_idx - 1]
                    approved = True
                else:
                    # Fallback to highest drift
                    best_edit = max(edits, key=lambda x: x.get('drift_score', 0.0)) if edits else None
                    approved = False
                
                best_edits.append({
                    'article_id': qa_group['article_id'],
                    'language': language,
                    'context': qa_group['context'],
                    'original_question': qa_group['original_question'],
                    'original_answer': qa_group['original_answer'],
                    'best_edit': best_edit,
                    'critic_approved': approved,
                    'critic_score': score,
                    'critic_reason': reason
                })
        
        # Save
        output_file = save_dir / f"{language}_best_edits.json"
        save_json(best_edits, output_file)
        print(f"Saved {len(best_edits)} best edits to {output_file}")
        
        return best_edits
    
    def select_all_best_edits(
        self,
        edits_by_language: Dict[str, List[Dict]],
        save_dir: Path
    ) -> Dict[str, List[Dict]]:
        """
        Select best edits for all languages.
        
        Args:
            edits_by_language: Dict mapping language to QA groups
            save_dir: Directory to save best edits
            
        Returns:
            Dict mapping language to best edits
        """
        all_best_edits = {}
        
        for lang, qa_groups in edits_by_language.items():
            best_edits = self.select_best_edits_for_language(lang, qa_groups, save_dir)
            all_best_edits[lang] = best_edits
        
        # Save summary
        total_approved = sum(
            sum(1 for edit in edits if edit['critic_approved'])
            for edits in all_best_edits.values()
        )
        total_edits = sum(len(edits) for edits in all_best_edits.values())
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'format_type': self.format_type,
            'languages': len(all_best_edits),
            'total_edits': total_edits,
            'total_approved': total_approved,
            'approval_rate': total_approved / total_edits if total_edits > 0 else 0.0
        }
        save_json(summary, save_dir / "summary.json")
        
        return all_best_edits


# ========== Fine-Tuning with LoRA ==========

class LoRAFineTuner:
    """Fine-tune model with LoRA on multi-task data"""
    
    def __init__(self, format_type: str):
        """
        Initialize LoRA fine-tuner.
        
        Args:
            format_type: 'no_edits', 'qa_only', or 'all_formats'
        """
        self.format_type = format_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"\n{'='*80}")
        print(f"Initializing LoRA Fine-Tuner")
        print(f"{'='*80}")
        print(f"Device: {self.device}")
        print(f"Model: {Config.MODEL_NAME}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_NAME,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=Config.LORA_R,
            lora_alpha=Config.LORA_ALPHA,
            target_modules=Config.LORA_TARGET_MODULES,
            lora_dropout=Config.LORA_DROPOUT,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
    def prepare_training_data(
        self,
        best_edits_by_language: Dict[str, List[Dict]]
    ) -> List[Dict]:
        """
        Prepare training data from best edits (multi-task, shuffled).
        
        Args:
            best_edits_by_language: Dict mapping language to best edits
            
        Returns:
            List of training examples (shuffled across all languages)
        """
        training_examples = []
        
        for lang, edits in best_edits_by_language.items():
            for edit in edits:
                # Get context (original or self-edited)
                if self.format_type == 'no_edits':
                    context = edit['context'][:500]
                else:
                    best_edit = edit.get('best_edit')
                    if best_edit and best_edit.get('generated_text'):
                        # Use generated text as enriched/modified context
                        # This allows implications, rewrites, chain-of-thought to enhance context
                        context = best_edit['generated_text'][:500]
                    else:
                        # Fallback to original
                        context = edit['context'][:500]
                
                # Always use original QA (the answer should actually answer the question!)
                question = edit['original_question']
                answer = edit['original_answer']
                
                # Create training text
                text = f"Context: {context}\nQuestion: {question}\nAnswer: {answer}"
                
                training_examples.append({
                    'text': text,
                    'language': lang
                })
        
        # Shuffle to mix all languages
        random.shuffle(training_examples)
        
        print(f"\nPrepared {len(training_examples)} training examples (shuffled across {len(best_edits_by_language)} languages)")
        
        return training_examples
    
    def create_dataset(self, examples: List[Dict]) -> Dataset:
        """
        Create Hugging Face dataset.
        
        Args:
            examples: List of training examples
            
        Returns:
            Hugging Face Dataset
        """
        def tokenize_function(examples_batch):
            return self.tokenizer(
                examples_batch['text'],
                truncation=True,
                max_length=Config.MAX_LENGTH,
                padding='max_length'
            )
        
        # Convert to dataset
        dataset = Dataset.from_list(examples)
        dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text', 'language'])
        
        return dataset
    
    def train(
        self,
        train_examples: List[Dict],
        val_examples: List[Dict] = None
    ) -> Dict[str, Any]:
        """
        Train model with LoRA.
        
        Args:
            train_examples: Training examples
            val_examples: Validation examples
            
        Returns:
            Training history
        """
        print(f"\n{'='*80}")
        print(f"Starting LoRA Fine-Tuning")
        print(f"{'='*80}")
        
        # Create datasets
        train_dataset = self.create_dataset(train_examples)
        val_dataset = self.create_dataset(val_examples) if val_examples else None
        
        # Training arguments
        output_dir = Config.CHECKPOINTS_DIR / f"{self.format_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=Config.NUM_EPOCHS,
            per_device_train_batch_size=Config.BATCH_SIZE,
            per_device_eval_batch_size=Config.BATCH_SIZE,
            gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION_STEPS,
            learning_rate=Config.LEARNING_RATE,
            warmup_steps=Config.WARMUP_STEPS,
            logging_steps=10,
            eval_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if val_dataset else False,
            save_total_limit=2,
            fp16=self.device == "cuda",
            report_to="none",
            remove_unused_columns=False
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator
        )
        
        # Train
        train_result = trainer.train()
        
        # Save final model
        final_model_path = Config.CHECKPOINTS_DIR / f"{self.format_type}_final"
        trainer.save_model(str(final_model_path))
        print(f"\nSaved final model to {final_model_path}")
        
        # Return training history
        return {
            'train_loss': train_result.training_loss,
            'train_runtime': train_result.metrics.get('train_runtime'),
            'train_samples_per_second': train_result.metrics.get('train_samples_per_second'),
            'output_dir': str(output_dir)
        }


# ========== Main Pipeline ==========

class SelfAdaptationPipeline:
    """End-to-end self-adaptation pipeline"""
    
    def __init__(self, format_type: str, samples_per_lang: int):
        """
        Initialize pipeline.
        
        Args:
            format_type: 'no_edits', 'qa_only', or 'all_formats'
            samples_per_lang: Number of samples per language
        """
        self.format_type = format_type
        self.samples_per_lang = samples_per_lang
        
        # Initialize components
        self.data_manager = DataManager(format_type, samples_per_lang)
        self.edit_generator = EditGenerator(format_type)
        self.critic_selector = CriticSelector(format_type)
        self.metrics_calculator = MetricsCalculator()
        
        print(f"\n{'='*80}")
        print(f"SELF-ADAPTATION PIPELINE")
        print(f"{'='*80}")
        print(f"Format: {format_type}")
        print(f"Samples per language: {samples_per_lang}")
        print(f"Languages: {', '.join(get_available_languages())}")
    
    def run(self):
        """Run complete pipeline."""
        
        # Step 1: Load data
        print(f"\n{'='*80}")
        print(f"STEP 1: Loading Data")
        print(f"{'='*80}")
        
        train_data = self.data_manager.load_data_for_all_languages(split="train")
        train_data, val_data = self.data_manager.prepare_training_validation_split(train_data)
        
        # Step 2: Generate self-edits
        print(f"\n{'='*80}")
        print(f"STEP 2: Generating Self-Edits")
        print(f"{'='*80}")
        
        train_edits = self.edit_generator.generate_all_edits(
            train_data,
            self.data_manager.edits_dir
        )
        
        # Step 3: Select best edits with critic
        print(f"\n{'='*80}")
        print(f"STEP 3: Selecting Best Edits with Critic")
        print(f"{'='*80}")
        
        best_train_edits = self.critic_selector.select_all_best_edits(
            train_edits,
            self.data_manager.best_edits_dir
        )
        
        # Step 4: Fine-tune with LoRA
        print(f"\n{'='*80}")
        print(f"STEP 4: Fine-Tuning with LoRA")
        print(f"{'='*80}")
        
        fine_tuner = LoRAFineTuner(self.format_type)
        
        train_examples = fine_tuner.prepare_training_data(best_train_edits)
        
        # For validation, we'll use original QA (no edits)
        val_examples = []
        for lang, articles in val_data.items():
            for article in articles:
                if not article['qa_pairs']:
                    continue
                qa = article['qa_pairs'][0]
                context = article['context'][:500]
                text = f"Context: {context}\nQuestion: {qa['question']}\nAnswer: {qa['answer']}"
                val_examples.append({'text': text, 'language': lang})
        
        training_history = fine_tuner.train(train_examples, val_examples)
        
        # Save training history
        save_json(
            training_history,
            Config.LOGS_DIR / f"{self.format_type}_training_history.json"
        )
        
        # Step 5: Evaluation
        print(f"\n{'='*80}")
        print(f"STEP 5: Evaluation")
        print(f"{'='*80}")
        
        self._evaluate_model(fine_tuner, val_data, best_train_edits)
        
        print(f"\n{'='*80}")
        print(f"PIPELINE COMPLETED!")
        print(f"{'='*80}")
        print(f"\nResults saved to: {Config.RESULTS_DIR}")
        print(f"  - Edits: {self.data_manager.edits_dir}")
        print(f"  - Best edits: {self.data_manager.best_edits_dir}")
        print(f"  - Checkpoints: {Config.CHECKPOINTS_DIR}")
        print(f"  - Metrics: {Config.METRICS_DIR}")
        print(f"  - Logs: {Config.LOGS_DIR}")
    
    def _evaluate_model(
        self,
        fine_tuner: 'LoRAFineTuner',
        val_data: Dict[str, List[Dict]],
        train_edits: Dict[str, List[Dict]]
    ):
        """
        Evaluate fine-tuned model on validation data.
        
        Args:
            fine_tuner: Fine-tuned model
            val_data: Validation data by language
            train_edits: Training edits for comparison
        """
        print("\nGenerating predictions on validation set...")
        
        # Collect predictions and ground truths by language
        results_by_language = {}
        
        for lang, articles in val_data.items():
            print(f"\nEvaluating {lang.upper()}...")
            
            predictions = []
            ground_truths = []
            contexts = []
            questions = []
            
            for article in tqdm(articles, desc=f"Generating for {lang}"):
                if not article['qa_pairs']:
                    continue
                
                qa = article['qa_pairs'][0]
                context = article['context'][:500]
                question = qa['question']
                answer = qa['answer']
                
                # Generate answer from fine-tuned model
                prediction = self._generate_answer(fine_tuner, context, question)
                
                predictions.append(prediction)
                ground_truths.append(answer)
                contexts.append(context)
                questions.append(question)
            
            if predictions:
                results_by_language[lang] = {
                    'predictions': predictions,
                    'ground_truths': ground_truths,
                    'contexts': contexts,
                    'questions': questions
                }
        
        # Calculate metrics
        print("\n" + "="*80)
        print("CALCULATING METRICS")
        print("="*80)
        
        language_metrics = self.metrics_calculator.calculate_language_metrics(results_by_language)
        
        # Calculate cross-lingual metrics
        cross_lingual_f1 = self.metrics_calculator.cross_lingual_f1(language_metrics)
        
        # For demonstration, consider first 3 languages as "seen" and rest as "unseen"
        all_languages = list(language_metrics.keys())
        seen_languages = all_languages[:3] if len(all_languages) > 3 else all_languages[:1]
        unseen_languages = all_languages[3:] if len(all_languages) > 3 else all_languages[1:]
        
        # Calculate transfer metrics
        zero_shot_score = self.metrics_calculator.zero_shot_transfer_score(
            language_metrics, unseen_languages
        ) if unseen_languages else 0.0
        
        transfer_gap = self.metrics_calculator.transfer_gap(
            language_metrics, seen_languages, unseen_languages
        ) if seen_languages and unseen_languages else {}
        
        # Generate comprehensive report
        report = self.metrics_calculator.generate_summary_report(
            language_metrics,
            baseline_metrics=None,  # Can add baseline comparison later
            seen_languages=seen_languages if seen_languages else None,
            unseen_languages=unseen_languages if unseen_languages else None
        )
        
        # Print report
        self.metrics_calculator.print_summary_table(report)
        
        # Save metrics
        metrics_file = Config.METRICS_DIR / f"{self.format_type}_metrics.json"
        save_json(report, metrics_file)
        print(f"\nMetrics saved to: {metrics_file}")
        
        # Save detailed results
        detailed_results = {
            'format_type': self.format_type,
            'timestamp': datetime.now().isoformat(),
            'language_results': results_by_language,
            'metrics': report,
            'training_summary': {
                'total_training_samples': sum(len(edits) for edits in train_edits.values()),
                'total_validation_samples': sum(len(r['predictions']) for r in results_by_language.values()),
                'languages': list(language_metrics.keys())
            }
        }
        
        detailed_file = Config.METRICS_DIR / f"{self.format_type}_detailed_results.json"
        save_json(detailed_results, detailed_file)
        print(f"Detailed results saved to: {detailed_file}")
        
        # Create summary CSV for easy viewing
        self._save_metrics_csv(language_metrics, report)
    
    def _generate_answer(
        self,
        fine_tuner: 'LoRAFineTuner',
        context: str,
        question: str,
        max_new_tokens: int = 100
    ) -> str:
        """
        Generate answer from fine-tuned model.
        
        Args:
            fine_tuner: Fine-tuned model with LoRA adapters
            context: Context passage
            question: Question to answer
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated answer string
        """
        # Format input prompt
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        
        # Tokenize
        inputs = fine_tuner.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=Config.MAX_LENGTH - max_new_tokens
        ).to(fine_tuner.device)
        
        # Generate
        with torch.no_grad():
            outputs = fine_tuner.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=fine_tuner.tokenizer.eos_token_id,
                eos_token_id=fine_tuner.tokenizer.eos_token_id
            )
        
        # Decode and extract answer
        full_response = fine_tuner.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated answer (after "Answer:")
        if "Answer:" in full_response:
            answer = full_response.split("Answer:")[-1].strip()
        else:
            answer = full_response[len(prompt):].strip()
        
        # Clean up - take only first sentence or line if multiple
        if '\n' in answer:
            answer = answer.split('\n')[0].strip()
        
        return answer
    
    def _save_metrics_csv(self, language_metrics: Dict, report: Dict):
        """Save metrics to CSV format for easy viewing."""
        import csv
        
        csv_file = Config.METRICS_DIR / f"{self.format_type}_metrics.csv"
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(['Language', 'EM', 'F1', 'Num_Samples'])
            
            # Language-wise metrics
            for lang, metrics in sorted(language_metrics.items()):
                writer.writerow([
                    lang,
                    f"{metrics['em']:.4f}",
                    f"{metrics['f1']:.4f}",
                    metrics['num_samples']
                ])
            
            # Overall metrics
            writer.writerow([])
            writer.writerow(['Overall Statistics', '', '', ''])
            writer.writerow(['Average EM', f"{report['overall']['avg_em']:.4f}", '', ''])
            writer.writerow(['Average F1', f"{report['overall']['avg_f1']:.4f}", '', ''])
            writer.writerow(['Std EM', f"{report['overall']['std_em']:.4f}", '', ''])
            writer.writerow(['Std F1', f"{report['overall']['std_f1']:.4f}", '', ''])
            
            # Cross-lingual metrics
            writer.writerow([])
            writer.writerow(['Cross-Lingual Metrics', '', '', ''])
            writer.writerow(['Cross-lingual F1', f"{report['cross_lingual_f1']:.4f}", '', ''])
            
            if 'zero_shot_transfer_score' in report:
                writer.writerow(['Zero-shot Transfer Score', f"{report['zero_shot_transfer_score']:.4f}", '', ''])
            
            if 'transfer_gap' in report:
                gap = report['transfer_gap']
                writer.writerow([])
                writer.writerow(['Transfer Gap Analysis', '', '', ''])
                writer.writerow(['Seen Performance', f"{gap['seen_performance']:.4f}", '', ''])
                writer.writerow(['Unseen Performance', f"{gap['unseen_performance']:.4f}", '', ''])
                writer.writerow(['EM Gap', f"{gap['em_gap']:.4f}", '', ''])
                writer.writerow(['F1 Gap', f"{gap['f1_gap']:.4f}", '', ''])
                writer.writerow(['Avg Gap', f"{gap['avg_gap']:.4f}", '', ''])
        
        print(f"Metrics CSV saved to: {csv_file}")


# ========== CLI ==========

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Self-Adaptation Pipeline for Multilingual Continual Learning'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['no_edits', 'qa_only', 'all_formats'],
        required=True,
        help='Format type: no_edits (baseline), qa_only (QA self-edits), all_formats (all 4 formats)'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=10,
        help='Number of samples per language (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(Config.RANDOM_SEED)
    
    # Ensure directories
    ensure_directories()
    
    # Run pipeline
    pipeline = SelfAdaptationPipeline(args.format, args.samples)
    pipeline.run()


if __name__ == "__main__":
    main()
