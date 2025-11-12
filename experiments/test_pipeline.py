#!/usr/bin/env python3
"""
Pipeline Component Testing

Test each step of the pipeline individually:
1. Edit Generation (with new validation)
2. Critic Selection (with format-specific criteria)
3. Context Transformation Fine-Tuning (new approach)

Usage:
    python test_pipeline.py --step [1|2|3] --samples 5
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, 'src')

def test_edit_generation(num_samples=5):
    """Test Step 1: Edit generation with validation"""
    print("\n" + "="*80)
    print("TESTING STEP 1: EDIT GENERATION WITH VALIDATION")
    print("="*80)
    
    from multi_format_self_edit import MultiFormatSelfEdit
    from data_loader import load_xquad_data
    from constants import LANGUAGES
    
    # Test on one language
    test_lang = "en"
    print(f"\nTesting on {test_lang}...")
    
    # Load data
    data = load_xquad_data(test_lang, max_samples=num_samples)
    print(f"Loaded {len(data)} samples")
    
    # Initialize generator
    generator = MultiFormatSelfEdit(
        model_name="Qwen/Qwen2.5-7B",
        cache_dir="./cache"
    )
    
    # Generate edits for all 4 formats
    results = {
        'language': test_lang,
        'samples_tested': num_samples,
        'formats': {}
    }
    
    for format_type in ['rewrite', 'implications', 'chain_of_thought', 'self_qa']:
        print(f"\n--- Testing {format_type} format ---")
        
        format_results = {
            'valid': 0,
            'invalid': 0,
            'validation_reasons': [],
            'examples': []
        }
        
        for i, item in enumerate(data[:num_samples], 1):
            print(f"\nSample {i}/{num_samples}")
            context = item['context']
            print(f"Context (first 100 chars): {context[:100]}...")
            
            edit = generator.generate_edit(context, format_type)
            
            is_valid = edit.get('is_valid', False)
            validation_msg = edit.get('validation_message', 'No message')
            generated = edit.get('generated_text', '')
            
            if is_valid:
                format_results['valid'] += 1
                print(f"✅ VALID")
                print(f"Generated (first 150 chars): {generated[:150]}...")
            else:
                format_results['invalid'] += 1
                format_results['validation_reasons'].append(validation_msg)
                print(f"❌ INVALID: {validation_msg}")
                print(f"Generated (first 150 chars): {generated[:150]}...")
            
            # Store example
            format_results['examples'].append({
                'context': context[:200] + "...",
                'generated': generated[:200] + "..." if generated else "",
                'is_valid': is_valid,
                'validation_message': validation_msg,
                'semantic_similarity': edit.get('semantic_similarity', 0.0)
            })
        
        # Calculate stats
        total = format_results['valid'] + format_results['invalid']
        valid_pct = (format_results['valid'] / total * 100) if total > 0 else 0
        
        print(f"\n{format_type} Summary:")
        print(f"  Valid: {format_results['valid']}/{total} ({valid_pct:.1f}%)")
        print(f"  Invalid: {format_results['invalid']}/{total}")
        
        if format_results['validation_reasons']:
            print(f"  Rejection reasons:")
            for reason in set(format_results['validation_reasons']):
                count = format_results['validation_reasons'].count(reason)
                print(f"    - {reason}: {count}")
        
        results['formats'][format_type] = format_results
    
    # Save results
    output_file = Path("results/test_edit_generation.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to {output_file}")
    
    # Overall summary
    print("\n" + "="*80)
    print("OVERALL EDIT GENERATION SUMMARY")
    print("="*80)
    print(f"\n{'Format':<20} {'Valid':<10} {'Invalid':<10} {'Valid %':<10}")
    print("-" * 50)
    
    for fmt, res in results['formats'].items():
        total = res['valid'] + res['invalid']
        pct = (res['valid'] / total * 100) if total > 0 else 0
        print(f"{fmt:<20} {res['valid']:<10} {res['invalid']:<10} {pct:<10.1f}%")
    
    return results


def test_critic_selection(num_samples=5):
    """Test Step 2: Critic with format-specific criteria"""
    print("\n" + "="*80)
    print("TESTING STEP 2: CRITIC WITH FORMAT-SPECIFIC CRITERIA")
    print("="*80)
    
    from multi_format_self_edit import MultiFormatSelfEdit
    from critic import Critic
    from data_loader import load_xquad_data
    
    test_lang = "en"
    print(f"\nTesting on {test_lang}...")
    
    # Load data
    data = load_xquad_data(test_lang, max_samples=num_samples)
    
    # Initialize
    generator = MultiFormatSelfEdit(
        model_name="Qwen/Qwen2.5-7B",
        cache_dir="./cache"
    )
    critic = Critic()
    
    # Generate all 4 formats for each sample
    all_edits = []
    
    for i, item in enumerate(data[:num_samples], 1):
        print(f"\nSample {i}/{num_samples}")
        context = item['context']
        
        # Generate 4 format candidates
        candidates = []
        for format_type in ['rewrite', 'implications', 'chain_of_thought', 'self_qa']:
            edit = generator.generate_edit(context, format_type)
            candidates.append({
                'format_type': format_type,
                'generated_text': edit.get('generated_text', ''),
                'is_valid': edit.get('is_valid', False),
                'validation_message': edit.get('validation_message', ''),
                'semantic_similarity': edit.get('semantic_similarity', 0.0)
            })
        
        all_edits.append({
            'context': context,
            'question': item.get('question', ''),
            'answer': item.get('answer', ''),
            'candidates': candidates
        })
    
    # Batch evaluate with critic
    print("\n--- Running Critic Evaluation ---")
    
    batch_contexts = [e['context'] for e in all_edits]
    batch_candidates = [e['candidates'] for e in all_edits]
    
    critic_results = critic.evaluate_language_batch(
        contexts=batch_contexts,
        edits_list=batch_candidates,
        language_code=test_lang,
        batch_size=num_samples
    )
    
    # Analyze results
    results = {
        'language': test_lang,
        'samples_tested': num_samples,
        'evaluations': []
    }
    
    print("\n" + "="*80)
    print("CRITIC EVALUATION RESULTS")
    print("="*80)
    
    approved_count = 0
    rejected_count = 0
    format_selections = {'rewrite': 0, 'implications': 0, 'chain_of_thought': 0, 'self_qa': 0}
    
    for i, (edit_data, critic_result) in enumerate(zip(all_edits, critic_results), 1):
        print(f"\nSample {i}:")
        print(f"Context: {edit_data['context'][:100]}...")
        
        best_format = critic_result.get('best_format', 'unknown')
        score = critic_result.get('score', 0.0)
        reasoning = critic_result.get('reasoning', 'No reasoning')
        approved = critic_result.get('approved', False)
        
        print(f"Best Format: {best_format}")
        print(f"Score: {score}/10")
        print(f"Approved: {'✅ YES' if approved else '❌ NO (< 6.0 threshold)'}")
        print(f"Reasoning: {reasoning[:150]}...")
        
        if approved:
            approved_count += 1
            format_selections[best_format] = format_selections.get(best_format, 0) + 1
        else:
            rejected_count += 1
        
        results['evaluations'].append({
            'context': edit_data['context'][:200] + "...",
            'best_format': best_format,
            'score': score,
            'approved': approved,
            'reasoning': reasoning
        })
    
    # Summary
    print("\n" + "="*80)
    print("CRITIC SUMMARY")
    print("="*80)
    print(f"Approved: {approved_count}/{num_samples} ({approved_count/num_samples*100:.1f}%)")
    print(f"Rejected: {rejected_count}/{num_samples} ({rejected_count/num_samples*100:.1f}%)")
    print(f"\nFormat Distribution (approved only):")
    for fmt, count in format_selections.items():
        if count > 0:
            print(f"  {fmt}: {count}")
    
    results['summary'] = {
        'approved': approved_count,
        'rejected': rejected_count,
        'format_distribution': format_selections
    }
    
    # Save
    output_file = Path("results/test_critic_selection.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to {output_file}")
    
    return results


def test_fine_tuning():
    """Test Step 3: Context transformation fine-tuning"""
    print("\n" + "="*80)
    print("TESTING STEP 3: CONTEXT TRANSFORMATION FINE-TUNING")
    print("="*80)
    
    print("\nTo test fine-tuning, run:")
    print("  python experiments/finetune_context_transformation.py --epochs 1 --samples 100")
    print("\nThis will:")
    print("  1. Load best edits with quality scores >= 6.0")
    print("  2. Create format-specific training pairs: (instruction + context) → transformed_context")
    print("  3. Train for 1 epoch on 100 samples (quick test)")
    print("  4. Evaluate transformation capability (semantic similarity)")
    print("\nExpected output:")
    print("  - Semantic similarity per format")
    print("  - Overall transformation quality")
    print("  - Saved model checkpoint")
    print("\nRun the command above to test!")


def main():
    parser = argparse.ArgumentParser(description="Test pipeline components")
    parser.add_argument('--step', type=int, choices=[1, 2, 3], required=True,
                        help='Pipeline step to test: 1=Edit Generation, 2=Critic, 3=Fine-tuning')
    parser.add_argument('--samples', type=int, default=5,
                        help='Number of samples to test (default: 5)')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("PIPELINE COMPONENT TESTING")
    print("="*80)
    print(f"Step: {args.step}")
    print(f"Samples: {args.samples}")
    
    if args.step == 1:
        test_edit_generation(num_samples=args.samples)
    elif args.step == 2:
        test_critic_selection(num_samples=args.samples)
    elif args.step == 3:
        test_fine_tuning()
    
    print("\n" + "="*80)
    print("TESTING COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
