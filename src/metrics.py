"""
Evaluation Metrics for Continual Learning

Provides comprehensive metrics for evaluating multilingual continual learning:
- Exact Match (EM)
- F1 Score
- Semantic Similarity
- Cross-lingual Transfer Metrics
- Training Curve Logging
"""

import re
import string
import numpy as np
from typing import List, Dict, Tuple, Any
from collections import defaultdict
from sentence_transformers import SentenceTransformer


class MetricsCalculator:
    """Calculate all evaluation metrics for continual learning"""
    
    def __init__(self, embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Initialize metrics calculator.
        
        Args:
            embedding_model: Sentence transformer model for semantic similarity
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.training_history = defaultdict(list)
    
    # ========== Basic QA Metrics ==========
    
    @staticmethod
    def normalize_answer(s: str) -> str:
        """
        Normalize answer text for comparison.
        
        Args:
            s: Answer string
            
        Returns:
            Normalized string
        """
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        
        def white_space_fix(text):
            return ' '.join(text.split())
        
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        
        def lower(text):
            return text.lower()
        
        return white_space_fix(remove_articles(remove_punc(lower(s))))
    
    def exact_match(self, prediction: str, ground_truth: str) -> float:
        """
        Calculate exact match score.
        
        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer
            
        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        return float(self.normalize_answer(prediction) == self.normalize_answer(ground_truth))
    
    def f1_score(self, prediction: str, ground_truth: str) -> float:
        """
        Calculate token-level F1 score.
        
        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer
            
        Returns:
            F1 score between 0.0 and 1.0
        """
        pred_tokens = self.normalize_answer(prediction).split()
        truth_tokens = self.normalize_answer(ground_truth).split()
        
        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return float(pred_tokens == truth_tokens)
        
        common_tokens = set(pred_tokens) & set(truth_tokens)
        
        if len(common_tokens) == 0:
            return 0.0
        
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(truth_tokens)
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def batch_em_f1(
        self,
        predictions: List[str],
        ground_truths: List[str]
    ) -> Tuple[float, float]:
        """
        Calculate average EM and F1 for a batch.
        
        Args:
            predictions: List of predicted answers
            ground_truths: List of ground truth answers
            
        Returns:
            Tuple of (average_em, average_f1)
        """
        em_scores = []
        f1_scores = []
        
        for pred, truth in zip(predictions, ground_truths):
            em_scores.append(self.exact_match(pred, truth))
            f1_scores.append(self.f1_score(pred, truth))
        
        return np.mean(em_scores), np.mean(f1_scores)
    
    # ========== Semantic Similarity ==========
    
    def semantic_similarity(
        self,
        texts1: List[str],
        texts2: List[str]
    ) -> float:
        """
        Calculate semantic similarity between two sets of texts.
        
        Args:
            texts1: First set of texts
            texts2: Second set of texts
            
        Returns:
            Average cosine similarity
        """
        if len(texts1) != len(texts2):
            raise ValueError("Text lists must have same length")
        
        embeddings1 = self.embedding_model.encode(texts1)
        embeddings2 = self.embedding_model.encode(texts2)
        
        similarities = []
        for emb1, emb2 in zip(embeddings1, embeddings2):
            similarity = np.dot(emb1, emb2) / (
                np.linalg.norm(emb1) * np.linalg.norm(emb2)
            )
            similarities.append(similarity)
        
        return float(np.mean(similarities))
    
    # ========== Cross-Lingual Transfer Metrics ==========
    
    def calculate_language_metrics(
        self,
        results_by_language: Dict[str, Dict[str, List[str]]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate EM and F1 for each language.
        
        Args:
            results_by_language: Dict mapping language to {
                'predictions': List[str],
                'ground_truths': List[str]
            }
            
        Returns:
            Dict mapping language to {'em': float, 'f1': float}
        """
        language_metrics = {}
        
        for lang, data in results_by_language.items():
            predictions = data['predictions']
            ground_truths = data['ground_truths']
            
            em, f1 = self.batch_em_f1(predictions, ground_truths)
            
            language_metrics[lang] = {
                'em': em,
                'f1': f1,
                'num_samples': len(predictions)
            }
        
        return language_metrics
    
    def cross_lingual_f1(
        self,
        language_metrics: Dict[str, Dict[str, float]]
    ) -> float:
        """
        Calculate average F1 across all languages.
        
        Args:
            language_metrics: Output from calculate_language_metrics
            
        Returns:
            Average F1 score across languages
        """
        f1_scores = [metrics['f1'] for metrics in language_metrics.values()]
        return float(np.mean(f1_scores))
    
    def zero_shot_transfer_score(
        self,
        language_metrics: Dict[str, Dict[str, float]],
        unseen_languages: List[str] = None
    ) -> float:
        """
        Calculate zero-shot transfer score for unseen languages.
        
        Args:
            language_metrics: Output from calculate_language_metrics
            unseen_languages: List of language codes considered "unseen"
                            If None, uses all languages
            
        Returns:
            Average of EM and F1 for unseen languages
        """
        if unseen_languages is None:
            unseen_languages = list(language_metrics.keys())
        
        em_scores = []
        f1_scores = []
        
        for lang in unseen_languages:
            if lang in language_metrics:
                em_scores.append(language_metrics[lang]['em'])
                f1_scores.append(language_metrics[lang]['f1'])
        
        if not em_scores:
            return 0.0
        
        avg_em = np.mean(em_scores)
        avg_f1 = np.mean(f1_scores)
        
        # Zero-shot transfer score is average of EM and F1
        return float((avg_em + avg_f1) / 2)
    
    def transfer_gap(
        self,
        language_metrics: Dict[str, Dict[str, float]],
        seen_languages: List[str],
        unseen_languages: List[str]
    ) -> Dict[str, float]:
        """
        Calculate performance gap between seen and unseen languages.
        
        Args:
            language_metrics: Output from calculate_language_metrics
            seen_languages: Languages emphasized during training
            unseen_languages: Languages not emphasized
            
        Returns:
            Dict with gap metrics
        """
        seen_em = np.mean([language_metrics[lang]['em'] for lang in seen_languages 
                          if lang in language_metrics])
        seen_f1 = np.mean([language_metrics[lang]['f1'] for lang in seen_languages 
                          if lang in language_metrics])
        
        unseen_em = np.mean([language_metrics[lang]['em'] for lang in unseen_languages 
                            if lang in language_metrics])
        unseen_f1 = np.mean([language_metrics[lang]['f1'] for lang in unseen_languages 
                            if lang in language_metrics])
        
        return {
            'em_gap': float(seen_em - unseen_em),
            'f1_gap': float(seen_f1 - unseen_f1),
            'avg_gap': float(((seen_em - unseen_em) + (seen_f1 - unseen_f1)) / 2),
            'seen_performance': float((seen_em + seen_f1) / 2),
            'unseen_performance': float((unseen_em + unseen_f1) / 2)
        }
    
    def relative_improvement(
        self,
        current_metrics: Dict[str, Dict[str, float]],
        baseline_metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate relative improvement over baseline per language.
        
        Args:
            current_metrics: Current model's language metrics
            baseline_metrics: Baseline model's language metrics
            
        Returns:
            Dict mapping language to improvement percentages
        """
        improvements = {}
        
        for lang in current_metrics:
            if lang not in baseline_metrics:
                continue
            
            curr_em = current_metrics[lang]['em']
            curr_f1 = current_metrics[lang]['f1']
            base_em = baseline_metrics[lang]['em']
            base_f1 = baseline_metrics[lang]['f1']
            
            # Avoid division by zero
            em_improvement = ((curr_em - base_em) / base_em * 100) if base_em > 0 else 0.0
            f1_improvement = ((curr_f1 - base_f1) / base_f1 * 100) if base_f1 > 0 else 0.0
            
            improvements[lang] = {
                'em_improvement_%': float(em_improvement),
                'f1_improvement_%': float(f1_improvement),
                'avg_improvement_%': float((em_improvement + f1_improvement) / 2)
            }
        
        return improvements
    
    # ========== Training Curve Logging ==========
    
    def log_training_step(
        self,
        step: int,
        epoch: int,
        loss: float,
        metrics: Dict[str, Any] = None
    ):
        """
        Log training step metrics.
        
        Args:
            step: Training step number
            epoch: Current epoch
            loss: Training loss
            metrics: Optional additional metrics (EM, F1, etc.)
        """
        entry = {
            'step': step,
            'epoch': epoch,
            'loss': loss
        }
        
        if metrics:
            entry.update(metrics)
        
        self.training_history['steps'].append(entry)
    
    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float = None,
        train_metrics: Dict[str, float] = None,
        val_metrics: Dict[str, float] = None
    ):
        """
        Log epoch-level metrics.
        
        Args:
            epoch: Epoch number
            train_loss: Training loss
            val_loss: Validation loss
            train_metrics: Training metrics (EM, F1, etc.)
            val_metrics: Validation metrics
        """
        entry = {
            'epoch': epoch,
            'train_loss': train_loss
        }
        
        if val_loss is not None:
            entry['val_loss'] = val_loss
        
        if train_metrics:
            entry['train_metrics'] = train_metrics
        
        if val_metrics:
            entry['val_metrics'] = val_metrics
        
        self.training_history['epochs'].append(entry)
    
    def get_training_history(self) -> Dict[str, List[Dict]]:
        """Get complete training history."""
        return dict(self.training_history)
    
    def reset_history(self):
        """Reset training history."""
        self.training_history = defaultdict(list)
    
    # ========== Summary Report ==========
    
    def generate_summary_report(
        self,
        language_metrics: Dict[str, Dict[str, float]],
        baseline_metrics: Dict[str, Dict[str, float]] = None,
        seen_languages: List[str] = None,
        unseen_languages: List[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive summary report.
        
        Args:
            language_metrics: Current model metrics
            baseline_metrics: Baseline metrics for comparison
            seen_languages: Languages emphasized during training
            unseen_languages: Languages not emphasized
            
        Returns:
            Complete summary report
        """
        report = {
            'language_metrics': language_metrics,
            'cross_lingual_f1': self.cross_lingual_f1(language_metrics)
        }
        
        # Zero-shot transfer
        if unseen_languages:
            report['zero_shot_transfer_score'] = self.zero_shot_transfer_score(
                language_metrics, unseen_languages
            )
        
        # Transfer gap
        if seen_languages and unseen_languages:
            report['transfer_gap'] = self.transfer_gap(
                language_metrics, seen_languages, unseen_languages
            )
        
        # Relative improvement
        if baseline_metrics:
            report['relative_improvement'] = self.relative_improvement(
                language_metrics, baseline_metrics
            )
        
        # Overall statistics
        all_em = [m['em'] for m in language_metrics.values()]
        all_f1 = [m['f1'] for m in language_metrics.values()]
        
        report['overall'] = {
            'avg_em': float(np.mean(all_em)),
            'avg_f1': float(np.mean(all_f1)),
            'std_em': float(np.std(all_em)),
            'std_f1': float(np.std(all_f1)),
            'min_em': float(np.min(all_em)),
            'max_em': float(np.max(all_em)),
            'min_f1': float(np.min(all_f1)),
            'max_f1': float(np.max(all_f1))
        }
        
        return report
    
    def print_summary_table(self, report: Dict[str, Any]):
        """
        Print formatted summary table.
        
        Args:
            report: Output from generate_summary_report
        """
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        
        # Language-wise metrics
        print("\nLanguage-wise Performance:")
        print(f"{'Language':<12} {'EM':<10} {'F1':<10} {'Samples':<10}")
        print("-" * 80)
        
        for lang, metrics in sorted(report['language_metrics'].items()):
            print(f"{lang:<12} {metrics['em']:<10.4f} {metrics['f1']:<10.4f} {metrics['num_samples']:<10}")
        
        # Overall metrics
        print("\n" + "-" * 80)
        overall = report['overall']
        print(f"{'Overall':<12} {overall['avg_em']:<10.4f} {overall['avg_f1']:<10.4f}")
        print(f"{'Std Dev':<12} {overall['std_em']:<10.4f} {overall['std_f1']:<10.4f}")
        
        # Cross-lingual metrics
        print("\n" + "="*80)
        print("Cross-Lingual Transfer Metrics:")
        print("-" * 80)
        print(f"Cross-lingual F1: {report['cross_lingual_f1']:.4f}")
        
        if 'zero_shot_transfer_score' in report:
            print(f"Zero-shot Transfer Score: {report['zero_shot_transfer_score']:.4f}")
        
        if 'transfer_gap' in report:
            gap = report['transfer_gap']
            print(f"\nTransfer Gap:")
            print(f"  Seen Performance: {gap['seen_performance']:.4f}")
            print(f"  Unseen Performance: {gap['unseen_performance']:.4f}")
            print(f"  EM Gap: {gap['em_gap']:.4f}")
            print(f"  F1 Gap: {gap['f1_gap']:.4f}")
            print(f"  Avg Gap: {gap['avg_gap']:.4f}")
        
        # Relative improvement
        if 'relative_improvement' in report:
            print("\n" + "="*80)
            print("Relative Improvement over Baseline (%):")
            print("-" * 80)
            print(f"{'Language':<12} {'EM Δ%':<12} {'F1 Δ%':<12} {'Avg Δ%':<12}")
            print("-" * 80)
            
            for lang, impr in sorted(report['relative_improvement'].items()):
                print(f"{lang:<12} {impr['em_improvement_%']:<12.2f} "
                      f"{impr['f1_improvement_%']:<12.2f} {impr['avg_improvement_%']:<12.2f}")
        
        print("="*80 + "\n")
