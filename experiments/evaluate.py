#!/usr/bin/env python3
"""
Evaluate Self-Edit Generation

Tests self-edit generation at scale across all languages.
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from native_language_self_edit import SelfEditGenerator
from english_self_edit import EnglishSelfEditGenerator
from data_loader import load_tydiqa_by_language, get_available_languages


class Evaluator:
    """Evaluate self-edit generation at scale"""
    
    def __init__(self, approach: str = "native_language", output_dir: str = None):
        """
        Initialize evaluator with specified approach.
        
        Args:
            approach: Either "native_language" or "english"
            output_dir: Output directory (auto-set based on approach if None)
        """
        self.approach = approach
        
        # Initialize appropriate generator
        if approach == "native_language":
            self.generator = SelfEditGenerator()
            self.output_dir = Path(output_dir) if output_dir else Path("results/native_language_edits")
        elif approach == "english":
            self.generator = EnglishSelfEditGenerator()
            self.output_dir = Path(output_dir) if output_dir else Path("results/english_edits")
        else:
            raise ValueError(f"Unknown approach: {approach}. Use 'native_language' or 'english'")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.all_results = []
        self.language_stats = {}
    
    def evaluate_language(
        self,
        language: str,
        n_samples: int = 20,
        n_edits_per_qa: int = 5,
        save_edits: bool = True
    ):
        """Evaluate self-edit generation for a language"""
        
        print(f"\n{'='*60}")
        print(f"Processing {language.upper()}")
        print(f"{'='*60}")
        
        # Load data
        articles = load_tydiqa_by_language(language, max_samples=n_samples)
        
        if not articles:
            print(f"No data available for {language}")
            return
        
        print(f"Loaded {len(articles)} articles")
        print(f"Generating {n_edits_per_qa} edits per QA pair...")
        
        results = []
        drift_scores = []
        
        for i, article in enumerate(articles):
            if not article['qa_pairs']:
                continue
            
            qa = article['qa_pairs'][0]
            
            # Generate multiple edits for this QA pair
            qa_edits = []
            for edit_num in range(n_edits_per_qa):
                result = self.generator.generate_edit(
                    context=article['context'],
                    question=qa['question'],
                    answer=qa['answer'],
                    language=language
                )
                result['edit_number'] = edit_num + 1
                qa_edits.append(result)
                drift_scores.append(result['drift_score'])
            
            # Store all edits for this QA
            qa_result = {
                'article_id': i,
                'context': article['context'],
                'original_question': qa['question'],
                'original_answer': qa['answer'],
                'language': language,
                'edits': qa_edits,
                'avg_drift': float(np.mean([e['drift_score'] for e in qa_edits])),
                'std_drift': float(np.std([e['drift_score'] for e in qa_edits]))
            }
            results.append(qa_result)
            
            # Progress indicator
            if (i + 1) % 5 == 0:
                print(f"  Processed {i + 1}/{len(articles)} articles ({(i+1)*n_edits_per_qa} total edits)")
        
        # Calculate statistics
        if drift_scores:
            avg_drift = np.mean(drift_scores)
            std_drift = np.std(drift_scores)
            min_drift = np.min(drift_scores)
            max_drift = np.max(drift_scores)
            
            stats = {
                "language": language,
                "n_samples": len(results),  # Number of QA pairs
                "n_edits_per_qa": n_edits_per_qa,
                "total_edits": len(drift_scores),  # Total edits generated
                "avg_drift": float(avg_drift),
                "std_drift": float(std_drift),
                "min_drift": float(min_drift),
                "max_drift": float(max_drift)
            }
            
            print(f"\nResults for {language.upper()}:")
            print(f"  QA pairs: {len(results)}")
            print(f"  Total edits: {len(drift_scores)}")
            print(f"  Average drift: {avg_drift:.4f} ({avg_drift*100:.2f}%)")
            print(f"  Std dev: {std_drift:.4f}")
            print(f"  Range: {min_drift:.4f} - {max_drift:.4f}")
            
            self.language_stats[language] = stats
            self.all_results.extend(results)
            
            # Save results
            if save_edits:
                output_file = self.output_dir / f"{language}_edits.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"  Saved to: {output_file}")
    
    def run_full_evaluation(self, n_samples_per_lang: int = 20):
        """Run evaluation across all languages"""
        
        print(f"\n{'='*60}")
        print("MULTILINGUAL SELF-EDIT EVALUATION")
        print(f"{'='*60}")
        print(f"Samples per language: {n_samples_per_lang}")
        print(f"Output directory: {self.output_dir}")
        
        languages = get_available_languages()
        
        for lang in languages:
            try:
                self.evaluate_language(lang, n_samples=n_samples_per_lang)
            except Exception as e:
                print(f"Error processing {lang}: {e}")
        
        # Summary
        self._print_summary()
        self._save_summary()
    
    def _print_summary(self):
        """Print overall summary"""
        
        if not self.language_stats:
            print("\nNo results to summarize")
            return
        
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        
        # Collect all drift scores from all edits
        all_drifts = []
        for result in self.all_results:
            for edit in result['edits']:
                all_drifts.append(edit['drift_score'])
        
        if all_drifts:
            print(f"\nOverall Statistics:")
            print(f"  Total QA pairs: {len(self.all_results)}")
            print(f"  Total edits: {len(all_drifts)}")
            print(f"  Languages: {len(self.language_stats)}")
            print(f"  Average drift: {np.mean(all_drifts):.4f} ({np.mean(all_drifts)*100:.2f}%)")
            print(f"  Std dev: {np.std(all_drifts):.4f}")
            
            print(f"\nPer-Language Drift:")
            for lang, stats in sorted(self.language_stats.items()):
                print(f"  {lang.upper()}: {stats['avg_drift']:.4f} ({stats['avg_drift']*100:.2f}%)")
    
    def _save_summary(self):
        """Save summary to JSON"""
        
        # Collect all drift scores
        all_drifts = []
        for result in self.all_results:
            for edit in result['edits']:
                all_drifts.append(edit['drift_score'])
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_qa_pairs": len(self.all_results),
            "total_edits": len(all_drifts),
            "languages": len(self.language_stats),
            "overall_drift": float(np.mean(all_drifts)) if all_drifts else 0.0,
            "language_stats": self.language_stats
        }
        
        summary_file = self.output_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate self-edit generation')
    parser.add_argument(
        '--approach',
        type=str,
        choices=['native_language', 'english'],
        default='native_language',
        help='Approach to use: native_language (X→X) or english (X→English)'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=20,
        help='Number of samples per language (default: 20)'
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Running evaluation with {args.approach.upper()} approach")
    print(f"{'='*60}\n")
    
    evaluator = Evaluator(approach=args.approach)
    evaluator.run_full_evaluation(n_samples_per_lang=args.samples)
