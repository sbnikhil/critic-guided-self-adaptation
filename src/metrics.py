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
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)

            if norm1 == 0 or norm2 == 0:
                # Skip invalid embeddings
                similarities.append(0.0)
                continue

            similarity = np.dot(emb1, emb2) / (norm1 * norm2)
            similarities.append(similarity)

        return float(np.mean(similarities))
    
    
    def calculate_language_metrics(
        self,
        results_by_language: Dict[str, Dict[str, List[str]]],
        semantic_similarities: Dict[str, float] = None,
        drift_scores: Dict[str, List[float]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate EM, F1, and semantic similarity for each language.
        
        Args:
            results_by_language: Dict mapping language to {
                'predictions': List[str],
                'ground_truths': List[str]
            }
            semantic_similarities: Optional dict mapping language to semantic similarity
        Returns:
            Dict mapping language to {'em': float, 'f1': float, 'semantic_similarity': float}
        """
        language_metrics = {}
        
        for lang, data in results_by_language.items():
            predictions = data['predictions']
            ground_truths = data['ground_truths']
            em, f1 = self.batch_em_f1(predictions, ground_truths)
            metrics = {
                'em': em,
                'f1': f1,
                'num_samples': len(predictions)
            }
            if semantic_similarities and lang in semantic_similarities:
                metrics['semantic_similarity'] = semantic_similarities[lang]
            # Drift/diversity metrics
            if drift_scores and lang in drift_scores and drift_scores[lang]:
                arr = np.array(drift_scores[lang])
                metrics['mean_drift'] = float(np.mean(arr))
                metrics['std_drift'] = float(np.std(arr))
                metrics['min_drift'] = float(np.min(arr))
                metrics['max_drift'] = float(np.max(arr))
            language_metrics[lang] = metrics
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
        all_sem = [m['semantic_similarity'] for m in language_metrics.values() if 'semantic_similarity' in m]
        all_drift = [m['mean_drift'] for m in language_metrics.values() if 'mean_drift' in m]
        # If there are no language metrics (no predictions), return safe defaults
        if len(all_em) == 0 or len(all_f1) == 0:
            report['overall'] = {
                'avg_em': 0.0,
                'avg_f1': 0.0,
                'std_em': 0.0,
                'std_f1': 0.0,
                'min_em': 0.0,
                'max_em': 0.0,
                'min_f1': 0.0,
                'max_f1': 0.0,
            }
        else:
            report['overall'] = {
                'avg_em': float(np.mean(all_em)),
                'avg_f1': float(np.mean(all_f1)),
                'std_em': float(np.std(all_em)),
                'std_f1': float(np.std(all_f1)),
                'min_em': float(np.min(all_em)),
                'max_em': float(np.max(all_em)),
                'min_f1': float(np.min(all_f1)),
                'max_f1': float(np.max(all_f1)),
            }
        if all_sem:
            report['overall']['avg_semantic_similarity'] = float(np.mean(all_sem))
            report['overall']['std_semantic_similarity'] = float(np.std(all_sem))
            report['overall']['min_semantic_similarity'] = float(np.min(all_sem))
            report['overall']['max_semantic_similarity'] = float(np.max(all_sem))
        if all_drift:
            report['overall']['avg_drift'] = float(np.mean(all_drift))
            report['overall']['std_drift'] = float(np.std(all_drift))
            report['overall']['min_drift'] = float(np.min(all_drift))
            report['overall']['max_drift'] = float(np.max(all_drift))
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
        print("\nLanguage-wise Performance (Multilingual Robustness):")
        print(f"{'Language':<12} {'EM':<10} {'F1':<10} {'SemSim':<10} {'Drift':<10} {'Samples':<10}")
        print("-" * 90)
        # Find worst-case languages
        min_f1_lang = None
        min_f1 = float('inf')
        min_sem_lang = None
        min_sem = float('inf')
        max_drift_lang = None
        max_drift = float('-inf')
        for lang, metrics in report['language_metrics'].items():
            if metrics.get('f1', float('inf')) < min_f1:
                min_f1 = metrics['f1']
                min_f1_lang = lang
            if metrics.get('semantic_similarity', float('inf')) < min_sem:
                min_sem = metrics['semantic_similarity']
                min_sem_lang = lang
            if metrics.get('mean_drift', float('-inf')) > max_drift:
                max_drift = metrics['mean_drift']
                max_drift_lang = lang
        for lang, metrics in sorted(report['language_metrics'].items()):
            semsim = metrics.get('semantic_similarity', float('nan'))
            drift = metrics.get('mean_drift', float('nan'))
            row = f"{lang:<12} {metrics['em']:<10.4f} {metrics['f1']:<10.4f} {semsim:<10.4f} {drift:<10.4f} {metrics['num_samples']:<10}"
            # Highlight worst-case
            if lang == min_f1_lang:
                row += "   <== LOWEST F1"
            if lang == min_sem_lang:
                row += "   <== LOWEST SemSim"
            if lang == max_drift_lang:
                row += "   <== HIGHEST Drift"
            print(row)
        # Overall metrics
        print("\n" + "-" * 80)
        overall = report['overall']
        print(f"{'Overall':<12} {overall['avg_em']:<10.4f} {overall['avg_f1']:<10.4f} {overall.get('avg_semantic_similarity', float('nan')):<10.4f} {overall.get('avg_drift', float('nan')):<10.4f}")
        print(f"{'Std Dev':<12} {overall['std_em']:<10.4f} {overall['std_f1']:<10.4f} {overall.get('std_semantic_similarity', float('nan')):<10.4f} {overall.get('std_drift', float('nan')):<10.4f}")
        if 'min_semantic_similarity' in overall or 'min_drift' in overall:
            print(f"{'Min':<12} {overall['min_em']:<10.4f} {overall['min_f1']:<10.4f} {overall.get('min_semantic_similarity', float('nan')):<10.4f} {overall.get('min_drift', float('nan')):<10.4f}")
            print(f"{'Max':<12} {overall['max_em']:<10.4f} {overall['max_f1']:<10.4f} {overall.get('max_semantic_similarity', float('nan')):<10.4f} {overall.get('max_drift', float('nan')):<10.4f}")
        else:
            print(f"{'Min':<12} {overall['min_em']:<10.4f} {overall['min_f1']:<10.4f}")
            print(f"{'Max':<12} {overall['max_em']:<10.4f} {overall['max_f1']:<10.4f}")
        print("\nSummary of Metrics:")
        print("- EM: Exact Match (QA correctness, strict)")
        print("- F1: Token-level F1 (QA overlap, partial credit)")
        print("- SemSim: Semantic Similarity (embedding-based, language-agnostic)")
        print("- Drift: Mean drift/diversity from original (higher = more diverse edits, but too high may mean off-topic)")
        print("- Std Dev: Diversity/variance across languages")
        print("- Min/Max: Best/worst language for each metric")
        print("\nMultilingual robustness is best when all languages have high F1, high SemSim, and moderate drift (not too low, not too high).\n")
        
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

    
    def evaluate_no_context_qa(
        self,
        model,
        tokenizer,
        questions: List[str],
        answers: List[str],
        languages: List[str],
        max_new_tokens: int = 100
    ) -> Dict[str, Any]:
        """
        Evaluate model on questions WITHOUT providing context.
        Tests if model internalized knowledge from fine-tuning (SEAL-style).
        
        Args:
            model: Fine-tuned model
            tokenizer: Model tokenizer
            questions: List of questions
            answers: List of ground truth answers
            languages: List of language codes
            max_new_tokens: Max tokens to generate
            
        Returns:
            Dict with detailed results per question and aggregated metrics
        """
        import torch
        
        results = []
        
        for question, answer, lang in zip(questions, answers, languages):
            # Generate answer WITHOUT context (key difference from standard QA)
            prompt = f"Question: {question}\nAnswer:"
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            predicted = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[-1]:], 
                skip_special_tokens=True
            ).strip()
            
            # Calculate metrics
            sem_sim = self.semantic_similarity([predicted], [answer])
            em = self.exact_match(predicted, answer)
            f1 = self.f1_score(predicted, answer)
            contains_ans = self._check_answer_presence(predicted, answer)
            
            results.append({
                'question': question,
                'predicted': predicted,
                'ground_truth': answer,
                'language': lang,
                'semantic_similarity': sem_sim,
                'exact_match': em,
                'f1': f1,
                'contains_answer': contains_ans
            })
        
        # Aggregate metrics
        aggregated = self._aggregate_no_context_results(results)
        
        return {
            'per_example': results,
            'aggregated': aggregated
        }
    
    def _check_answer_presence(self, predicted: str, ground_truth: str) -> bool:
        """Check if ground truth answer appears in prediction (normalized)."""
        pred_norm = self.normalize_answer(predicted)
        truth_norm = self.normalize_answer(ground_truth)
        
        # Check if GT is substring of prediction
        if truth_norm in pred_norm:
            return True
        
        # Check if they share significant tokens
        pred_tokens = set(pred_norm.split())
        truth_tokens = set(truth_norm.split())
        
        if len(truth_tokens) == 0:
            return False
        
        # If >50% of truth tokens appear in prediction
        overlap = len(pred_tokens & truth_tokens) / len(truth_tokens)
        return overlap >= 0.5
    
    def _aggregate_no_context_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Aggregate no-context QA results by language and overall."""
        by_language = defaultdict(list)
        
        for r in results:
            by_language[r['language']].append(r)
        
        language_metrics = {}
        for lang, lang_results in by_language.items():
            language_metrics[lang] = {
                'semantic_similarity': float(np.mean([r['semantic_similarity'] for r in lang_results])),
                'exact_match': float(np.mean([r['exact_match'] for r in lang_results])),
                'f1': float(np.mean([r['f1'] for r in lang_results])),
                'answer_presence': float(np.mean([r['contains_answer'] for r in lang_results])),
                'num_samples': len(lang_results)
            }
        
        # Overall metrics
        overall = {
            'semantic_similarity': float(np.mean([r['semantic_similarity'] for r in results])),
            'exact_match': float(np.mean([r['exact_match'] for r in results])),
            'f1': float(np.mean([r['f1'] for r in results])),
            'answer_presence': float(np.mean([r['contains_answer'] for r in results])),
            'num_samples': len(results)
        }
        
        # Cross-lingual metrics
        lang_sims = [m['semantic_similarity'] for m in language_metrics.values()]
        lang_f1s = [m['f1'] for m in language_metrics.values()]
        
        # 1. Gap between best and worst (lower is better - more consistent)
        cross_lingual_gap = max(lang_sims) - min(lang_sims) if lang_sims else 0.0
        
        # 2. Standard deviation across languages (lower is better - more consistent)
        cross_lingual_std = float(np.std(lang_sims)) if lang_sims else 0.0
        
        # 3. Coefficient of variation (normalized consistency metric)
        cross_lingual_cv = (cross_lingual_std / np.mean(lang_sims)) if lang_sims and np.mean(lang_sims) > 0 else 0.0
        
        # 4. Multilingual capability score (higher is better)
        # Average of all languages (rewards doing well across all languages)
        multilingual_score = float(np.mean(lang_sims)) if lang_sims else 0.0
        
        # 5. Low-resource language performance
        # Identify languages with smallest training data or lowest baseline
        # For TyDiQA, low-resource: sw, te, bn (compared to en)
        low_resource_langs = ['sw', 'te', 'bn']
        low_resource_results = [r for r in results if r['language'] in low_resource_langs]
        low_resource_score = float(np.mean([r['semantic_similarity'] for r in low_resource_results])) if low_resource_results else 0.0
        
        # 6. High-resource language performance (for comparison)
        high_resource_langs = ['en', 'ar', 'ru']
        high_resource_results = [r for r in results if r['language'] in high_resource_langs]
        high_resource_score = float(np.mean([r['semantic_similarity'] for r in high_resource_results])) if high_resource_results else 0.0
        
        # 7. Transfer gap (difference between high and low resource)
        transfer_gap = high_resource_score - low_resource_score
        
        return {
            'by_language': language_metrics,
            'overall': overall,
            'cross_lingual_gap': cross_lingual_gap,
            'cross_lingual_std': cross_lingual_std,
            'cross_lingual_cv': cross_lingual_cv,
            'multilingual_score': multilingual_score,
            'low_resource_score': low_resource_score,
            'high_resource_score': high_resource_score,
            'transfer_gap': transfer_gap
        }

