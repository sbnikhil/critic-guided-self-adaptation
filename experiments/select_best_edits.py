#!/usr/bin/env python3

import sys
import json
import argparse
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np
from dotenv import load_dotenv
from critic import score_edits_with_critic, compute_quality_scores

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from critic import (
    CriticScorer,
    QualityScorer,
    EditFilter,
    SamplingWeightCalculator,
    process_edits_pipeline
)
from data_loader import get_available_languages


def load_edits_from_json(filepath: Path) -> list:
    """Load edits from a language-specific JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def flatten_edits(articles: list, language: str) -> list:
    """
    Flatten article structure to list of edits.
    
    Input: [{"article_id": 0, "context": "...", "edits": [...]}, ...]
    Output: [{"article_id": 0, "original_context": "...", "language": "te", ...}, ...]
    """
    flat_edits = []
    
    for article in articles:
        article_id = article.get("article_id", 0)
        context = article.get("context", "")
        
        for edit in article.get("edits", []):
            # Add article-level fields to each edit
            edit["article_id"] = article_id
            edit["original_context"] = context
            edit["language"] = language
            flat_edits.append(edit)
    
    return flat_edits


def unflatten_edits(flat_edits: list) -> list:
    """
    Convert flattened edits back to article structure.
    Groups edits by article_id and reconstructs original structure.
    """
    article_groups = defaultdict(lambda: {"edits": [], "context": ""})
    
    for edit in flat_edits:
        article_id = edit.get("article_id", 0)
        
        # Extract article-level fields
        if not article_groups[article_id]["context"]:
            article_groups[article_id]["context"] = edit.get("original_context", "")
            article_groups[article_id]["article_id"] = article_id
            article_groups[article_id]["language"] = edit.get("language", "")
        
        # Remove article-level fields from edit before adding
        edit_copy = edit.copy()
        edit_copy.pop("article_id", None)
        edit_copy.pop("original_context", None)
        
        article_groups[article_id]["edits"].append(edit_copy)
    
    # Convert to list sorted by article_id
    articles = [article_groups[aid] for aid in sorted(article_groups.keys())]
    return articles


def main():
    parser = argparse.ArgumentParser(description='Run critic-based processing pipeline (Steps C-F)')
    parser.add_argument(
        '--input-folder',
        type=str,
        required=True,
        help='Input folder with edits (e.g., results/multi_format_all)'
    )
    parser.add_argument(
        '--output-folder',
        type=str,
        help='Output folder for filtered edits (default: results/best_edits/<input_folder_name>)'
    )
    parser.add_argument(
        '--languages',
        nargs='+',
        help='Languages to process (default: all found in input folder)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        help='Max articles to process per language (for testing)'
    )
    parser.add_argument(
        '--max-per-article',
        type=int,
        default=4,
        help='Max edits to score per article (default: 4 for cost control)'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=2.0,
        help='Weighting exponent for sampling (default: 2.0 = quadratic)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=5,
        help='Number of edits to score per API call (default: 5, reduces API calls by 5x)'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='Google API key for Gemini (or set GOOGLE_API_KEY env var)'
    )
    
    args = parser.parse_args()
    
    # Check API key
    api_key = args.api_key or os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        print("Error: Google API key required (set GOOGLE_API_KEY or use --api-key)")
        sys.exit(1)
    
    # Setup paths
    input_folder = Path(args.input_folder)
    
    if not input_folder.exists():
        print(f"Error: Input folder not found: {input_folder}")
        sys.exit(1)
    
    # Determine output folder
    if args.output_folder:
        output_folder = Path(args.output_folder)
    else:
        folder_name = input_folder.name
        output_folder = Path("results/best_edits") / folder_name
    
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Find language files
    edit_files = list(input_folder.glob("*_edits.json"))
    
    if not edit_files:
        print(f"Error: No *_edits.json files found in {input_folder}")
        sys.exit(1)
    
    # Filter to specified languages if provided
    if args.languages:
        edit_files = [f for f in edit_files if f.stem.replace("_edits", "") in args.languages]
    
    if not edit_files:
        print(f"Error: No matching language files found")
        sys.exit(1)
    
    print("=" * 70)
    print("CRITIC-BASED EDIT PROCESSING PIPELINE (Steps C-F)")
    print("=" * 70)
    print(f"Input:  {input_folder}")
    print(f"Output: {output_folder}")
    print(f"Languages: {[f.stem.replace('_edits', '') for f in edit_files]}")
    print(f"Max per article: {args.max_per_article} edits")
    print(f"Batch size: {args.batch_size} edits/call")
    print(f"Weighting: alpha = {args.alpha}")
    print("=" * 70)
    
    # Initialize critic once (reuse across languages)
    print("\nInitializing Gemini critic...")
    critic = CriticScorer(api_key=api_key)
    
    # Process each language
    summary = {
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "max_per_article": args.max_per_article,
            "batch_size": args.batch_size,
            "alpha": args.alpha
        },
        "languages": {}
    }
    
    for edit_file in edit_files:
        language = edit_file.stem.replace("_edits", "")
        
        print(f"\n{'='*70}")
        print(f"PROCESSING: {language.upper()}")
        print(f"{'='*70}")
        
        # Load edits
        print(f"Loading: {edit_file.name}")
        articles = load_edits_from_json(edit_file)
        
        if args.max_samples and len(articles) > args.max_samples:
            print(f"Limiting to {args.max_samples} articles (from {len(articles)})")
            articles = articles[:args.max_samples]
        
        # Flatten for processing
        print(f"Loaded {len(articles)} articles")
        flat_edits = flatten_edits(articles, language)
        print(f"Total edits: {len(flat_edits)}")
        
        # Run ONLY scoring + Q(e) computation (not filtering yet)
        # This gives us ALL edits with scores

        
        scored_edits = score_edits_with_critic(
            edits=flat_edits,
            max_per_article=args.max_per_article,
            batch_size=args.batch_size,
            critic=critic,
            verbose=True
        )
        
        scored_edits = compute_quality_scores(scored_edits)
        
        # Save ALL scored edits (including rejected ones)
        all_scored_articles = unflatten_edits(scored_edits)
        all_scored_file = output_folder / f"{language}_all_scored_edits.json"
        with open(all_scored_file, 'w', encoding='utf-8') as f:
            json.dump(all_scored_articles, f, indent=2, ensure_ascii=False)
        print(f"Saved all scored edits: {all_scored_file}")
        
        # Now run complete pipeline for filtered results
        from critic import filter_and_weight_edits
        
        processed_edits = filter_and_weight_edits(
            edits=scored_edits,
            alpha=args.alpha,
            verbose=True
        )
        
        # DEBUG: Check for critic errors
        error_count = sum(1 for e in scored_edits if 'critic_error' in e)
        if error_count > 0:
            print(f"\nWARNING: {error_count} edits had critic errors!")
            # Show first error
            for e in scored_edits:
                if 'critic_error' in e:
                    print(f"   Error: {e['critic_error']}")
                    break
        
        # Unflatten back to article structure (filtered only)
        filtered_articles = unflatten_edits(processed_edits)
        
        # Save results
        output_file = output_folder / f"{language}_best_edits.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_articles, f, indent=2, ensure_ascii=False)
        
        print(f"Saved: {output_file}")
        
        # Compute statistics
        total_scored = len(scored_edits)
        total_filtered = sum(len(a["edits"]) for a in filtered_articles)
        num_approved = sum(1 for e in scored_edits if e.get("critic_approved", False))
        num_rejected = total_scored - num_approved
        
        if total_filtered > 0:
            avg_quality = np.mean([
                e.get("quality_score", 0.0) 
                for a in filtered_articles 
                for e in a["edits"]
            ])
            avg_weight = np.mean([
                e.get("sampling_weight", 0.0) 
                for a in filtered_articles 
                for e in a["edits"]
            ])
            avg_critic_score = np.mean([
                e.get("critic_score", 0.0) 
                for a in filtered_articles 
                for e in a["edits"]
            ])
        else:
            avg_quality = avg_weight = avg_critic_score = 0.0
        
        # Statistics on rejected edits
        rejected_edits = [e for e in scored_edits if not e.get("critic_approved", False)]
        if rejected_edits:
            avg_rejected_score = np.mean([e.get("critic_score", 0) for e in rejected_edits])
            rejection_reasons = defaultdict(int)
            for e in rejected_edits:
                reason = e.get("critic_reason", "unknown")[:50]  # First 50 chars
                rejection_reasons[reason] += 1
        else:
            avg_rejected_score = 0.0
            rejection_reasons = {}
        
        summary["languages"][language] = {
            "articles": len(filtered_articles),
            "total_edits_scored": total_scored,
            "total_approved": num_approved,
            "total_rejected": num_rejected,
            "total_edits_after_filtering": total_filtered,
            "avg_quality_score": float(avg_quality),
            "avg_sampling_weight": float(avg_weight),
            "avg_critic_score_approved": float(avg_critic_score),
            "avg_critic_score_rejected": float(avg_rejected_score)
        }
        
        print(f"\n{language.upper()} Summary:")
        print(f"   Total scored: {total_scored} edits")
        print(f"   Approved: {num_approved} ({100*num_approved/total_scored:.1f}%)")
        print(f"   Rejected: {num_rejected} ({100*num_rejected/total_scored:.1f}%)")
        print(f"   After filtering: {total_filtered} edits")
        print(f"   Avg Q(e): {avg_quality:.3f}")
        print(f"   Avg weight: {avg_weight:.3f}")
        print(f"   Avg critic score (approved): {avg_critic_score:.1f}/10")
        if rejected_edits:
            print(f"   Avg critic score (rejected): {avg_rejected_score:.1f}/10")
    
    # Save overall summary
    summary_file = output_folder / "summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*70}")
    print("COMPLETE!")
    print(f"{'='*70}")
    print(f"Summary saved: {summary_file}")
    
    # Overall stats
    total_scored = sum(s["total_edits_scored"] for s in summary["languages"].values())
    total_approved = sum(s["total_approved"] for s in summary["languages"].values())
    total_rejected = sum(s["total_rejected"] for s in summary["languages"].values())
    total_after_filtering = sum(s["total_edits_after_filtering"] for s in summary["languages"].values())
    
    print(f"\nOverall Statistics:")
    print(f"   Languages processed: {len(summary['languages'])}")
    print(f"   Total scored: {total_scored}")
    print(f"   Total approved: {total_approved} ({100*total_approved/total_scored:.1f}%)")
    print(f"   Total rejected: {total_rejected} ({100*total_rejected/total_scored:.1f}%)")
    print(f"   After filtering: {total_after_filtering}")
    if total_scored > 0:
        print(f"   Overall retention: {100 * total_after_filtering / total_scored:.1f}%")

if __name__ == "__main__":
    main()
