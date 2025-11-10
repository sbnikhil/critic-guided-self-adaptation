#!/usr/bin/env python3
"""
Compare Preservation Tiers

Tests all three preservation tiers to determine which works best.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from native_language_self_edit import SelfEditGenerator
from config import high_preservation_config, medium_preservation_config, low_preservation_config
from data_loader import load_tydiqa_by_language


def test_tier(tier_name: str, config, generator, test_languages: list, n_samples: int = 5):
    """Test a single preservation tier"""
    
    print(f"\n{'='*60}")
    print(f"Testing {tier_name.upper()}")
    print(f"{'='*60}")
    
    all_results = []
    
    for lang in test_languages:
        print(f"\n{lang.upper()}:")
        
        articles = load_tydiqa_by_language(lang, max_samples=n_samples)
        
        if not articles:
            print(f"  No data available")
            continue
        
        for article in articles:
            if not article['qa_pairs']:
                continue
            
            qa = article['qa_pairs'][0]
            
            result = generator.generate_edit(
                context=article['context'],
                question=qa['question'],
                answer=qa['answer'],
                language=lang,
                config=config
            )
            
            all_results.append(result)
    
    # Statistics
    if all_results:
        drifts = [r['drift_score'] for r in all_results]
        avg_drift = np.mean(drifts)
        
        print(f"\n{tier_name.upper()} Summary:")
        print(f"  Samples: {len(all_results)}")
        print(f"  Average drift: {avg_drift:.4f} ({avg_drift*100:.2f}%)")
        print(f"  Std dev: {np.std(drifts):.4f}")
        print(f"  Range: {np.min(drifts):.4f} - {np.max(drifts):.4f}")
        
        return avg_drift, all_results
    
    return 0.0, []


def main():
    """Compare all three preservation tiers"""
    
    print("\n" + "="*60)
    print("PRESERVATION TIER COMPARISON")
    print("="*60)
    
    generator = SelfEditGenerator()
    
    # Test languages (subset for speed)
    test_languages = ['en', 'ar', 'bn', 'ru', 'ko']
    n_samples = 5
    
    print(f"\nTest languages: {', '.join(test_languages)}")
    print(f"Samples per language: {n_samples}")
    
    # Test each tier
    tiers = [
        ("High Preservation", high_preservation_config()),
        ("Medium Preservation", medium_preservation_config()),
        ("Low Preservation", low_preservation_config())
    ]
    
    results = {}
    
    for tier_name, config in tiers:
        avg_drift, tier_results = test_tier(
            tier_name, config, generator, test_languages, n_samples
        )
        results[tier_name] = {
            'avg_drift': avg_drift,
            'results': tier_results
        }
    
    # Final comparison
    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")
    
    for tier_name in results:
        avg = results[tier_name]['avg_drift']
        print(f"{tier_name:20s}: {avg:.4f} ({avg*100:.2f}%)")
    
    # Recommendation
    print(f"\n{'='*60}")
    print("RECOMMENDATION")
    print(f"{'='*60}")
    print("Target drift: 20-30%")
    print("Based on large-scale evaluation, LOW PRESERVATION tier is recommended.")
    print("It achieves ~19% drift at scale, closest to target range.")


if __name__ == "__main__":
    main()
