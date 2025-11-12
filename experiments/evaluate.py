#!/usr/bin/env python3

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multi_format_self_edit import MultiFormatSelfEditGenerator
from data_loader import load_tydiqa_by_language, get_available_languages


class Evaluator:
    
    def __init__(self, output_dir: str = None, format_type: str = None):
        self.format_type = format_type
        self.generator = MultiFormatSelfEditGenerator()
        
        if format_type == "self_qa":
            self.output_dir = Path(output_dir) if output_dir else Path("results/multi_format_qa")
        else:
            self.output_dir = Path(output_dir) if output_dir else Path("results/multi_format_all")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.all_results = []
        self.language_stats = {}
    
    def evaluate_language(
        self,
        language: str,
        n_samples: int = 20,
        n_edits_per_qa: int = None,
        save_edits: bool = True
    ):
        
        if n_edits_per_qa is None:
            if self.format_type == "self_qa":
                n_edits_per_qa = 5
            elif not self.format_type:
                n_edits_per_qa = 4
            else:
                n_edits_per_qa = 5
        
        print(f"\n{'='*60}")
        print(f"Processing {language.upper()}")
        print(f"{'='*60}")
        
        # Load data
        articles = load_tydiqa_by_language(language, max_samples=n_samples)
        
        if not articles:
            print(f"No data available for {language}")
            return
        
        print(f"Loaded {len(articles)} articles")
        print(f"Generating {n_edits_per_qa} edits per context (SEAL-style)...")
        
        results = []
        drift_scores = []
        
        for i, article in enumerate(articles):
            # SEAL approach: context → synthetic data (no Q&A needed)
            
            context_edits = []
            for edit_num in range(n_edits_per_qa):
                if self.format_type:
                    # Single format mode
                    result = self.generator.generate_edit(
                        context=article['context'],
                        language=language,
                        format_type=self.format_type
                    )
                else:
                    # Cycle through all 4 formats (SEAL-style)
                    all_formats = list(self.generator.GENERATION_FORMATS.keys())
                    format_idx = edit_num % len(all_formats)
                    result = self.generator.generate_edit(
                        context=article['context'],
                        language=language,
                        format_type=all_formats[format_idx]
                    )
                
                result['edit_number'] = edit_num + 1
                context_edits.append(result)
                drift_scores.append(result['drift_score'])
            
            # Store all edits for this context
            context_result = {
                'article_id': i,
                'context': article['context'],
                'language': language,
                'edits': context_edits,
                'avg_drift': float(np.mean([e['drift_score'] for e in context_edits])),
                'std_drift': float(np.std([e['drift_score'] for e in context_edits]))
            }
            results.append(context_result)
            
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
                "n_contexts": len(results),  # Number of contexts/passages
                "n_edits_per_context": n_edits_per_qa,
                "total_edits": len(drift_scores),  # Total synthetic edits generated
                "avg_drift": float(avg_drift),
                "std_drift": float(std_drift),
                "min_drift": float(min_drift),
                "max_drift": float(max_drift)
            }
            
            print(f"\nResults for {language.upper()}:")
            print(f"  Contexts: {len(results)}")
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
            "total_contexts": len(self.all_results),
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
        '--qa-only',
        action='store_true',
        help='Generate only self-QA format (default: all 4 formats)'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=20,
        help='Number of samples per language (default: 20)'
    )
    
    args = parser.parse_args()
    
    format_type = 'self_qa' if args.qa_only else None
    
    print(f"\n{'='*60}")
    if args.qa_only:
        print("Running MULTI-FORMAT self-edit evaluation (self-QA only)")
    else:
        print("Running MULTI-FORMAT self-edit evaluation (all 4 formats)")
    print(f"{'='*60}\n")
    
    evaluator = Evaluator(format_type=format_type)
    evaluator.run_full_evaluation(n_samples_per_lang=args.samples)
