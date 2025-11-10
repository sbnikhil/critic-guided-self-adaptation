#!/usr/bin/env python3

import sys
import json
import argparse
import os
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from critic import Critic
from data_loader import get_available_languages


def load_edits_from_folder(folder_path: Path) -> dict:
    
    edits_by_language = {}
    
    for lang_file in folder_path.glob("*_edits.json"):
        lang = lang_file.stem.replace("_edits", "")
        
        with open(lang_file, 'r', encoding='utf-8') as f:
            edits_by_language[lang] = json.load(f)
    
    return edits_by_language


def main():
    parser = argparse.ArgumentParser(description='Select best edits using Gemini critic')
    parser.add_argument(
        '--input-folder',
        type=str,
        required=True,
        help='Input folder with edits (e.g., results/multi_format_qa or results/multi_format_all)'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='Google API key for Gemini (or set GOOGLE_API_KEY env var)'
    )
    
    args = parser.parse_args()
    
    api_key = args.api_key or os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        print("Error: Google API key required")
        sys.exit(1)
    
    print("Initializing Gemini critic...")
    critic = Critic(api_key=api_key)
    
    input_folder = Path(args.input_folder)
    if not input_folder.exists():
        print(f"Error: Input folder not found: {input_folder}")
        sys.exit(1)
    
    print(f"\nLoading edits from: {input_folder}")
    edits_by_language = load_edits_from_folder(input_folder)
    
    if not edits_by_language:
        print("Error: No edit files found in input folder")
        sys.exit(1)
    
    print(f"Found edits for {len(edits_by_language)} languages")
    
    folder_name = input_folder.name
    output_folder = Path("results/best_edits") / folder_name
    output_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"Output folder: {output_folder}")
    
    all_best_edits = {}
    summary_stats = {
        'timestamp': datetime.now().isoformat(),
        'input_folder': str(input_folder),
        'languages_processed': 0,
        'total_qa_pairs': 0,
        'total_edits_evaluated': 0,
        'approved_rate': 0.0,
        'average_score': 0.0,
        'per_language': {}
    }
    
    for lang, qa_groups in edits_by_language.items():
        print(f"\n{'='*60}")
        print(f"Processing {lang.upper()}: {len(qa_groups)} QA pairs")
        print(f"{'='*60}")
        
        best_edits = []
        approved_count = 0
        scores = []
        total_edits = 0
        
        batch_size = 2
        print(f"Evaluating {lang} in batches of {batch_size} QA pairs")
        
        for batch_start in range(0, len(qa_groups), batch_size):
            batch_end = min(batch_start + batch_size, len(qa_groups))
            batch_qa = qa_groups[batch_start:batch_end]
            
            if batch_start > 0:
                time.sleep(6.5)
            
            print(f"  Batch {batch_start//batch_size + 1}: QA {batch_start+1}-{batch_end}")
            batch_results = critic.evaluate_language_batch(batch_qa)

            for i, (qa_group, br) in enumerate(zip(batch_qa, batch_results)):
                selected_idx = int(br.get('selected_index', -1))
                score = float(br.get('score', 0.0))
                reason = br.get('reason', '')
                raw = br.get('raw_response', '')

                edits = qa_group.get('edits', [])
                if 1 <= selected_idx <= len(edits):
                    best_edit = edits[selected_idx - 1]
                    approved = True if score > 0.0 else False
                    best_edit_number = best_edit.get('edit_number', selected_idx)
                else:
                    best_edit = max(edits, key=lambda x: x.get('drift_score', 0.0)) if edits else {}
                    approved = False
                    best_edit_number = best_edit.get('edit_number', -1)

                evaluations = []
                for j, edit in enumerate(edits, 1):
                    evaluations.append({
                        'edit_number': edit.get('edit_number', j),
                        'approved': (j == selected_idx),
                        'reason': reason if j == selected_idx else '',
                        'score': score if j == selected_idx else 0.0,
                        'raw_response': raw,
                        'drift_score': edit.get('drift_score', 0.0),
                        'edit': edit
                    })

                best = {
                    'article_id': qa_group.get('article_id', 0),
                    'language': lang,
                    'context': qa_group.get('context', ''),
                    'original_question': qa_group.get('original_question', ''),
                    'original_answer': qa_group.get('original_answer', ''),
                    'best_edit': best_edit,
                    'best_edit_number': best_edit_number,
                    'critic_approved': approved,
                    'critic_reason': reason,
                    'critic_score': score,
                    'all_evaluations': evaluations,
                    'total_edits': len(edits)
                }

                best_edits.append(best)
                if best['critic_approved']:
                    approved_count += 1
                scores.append(best['critic_score'])
                total_edits += best['total_edits']
        
        output_file = output_folder / f"{lang}_best_edits.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(best_edits, f, indent=2, ensure_ascii=False)
        
        log_file = output_folder / f"{lang}_detailed_log.txt"
        with open(log_file, 'w', encoding='utf-8') as lf:
            lf.write(f"Detailed critic log for {lang}\nGenerated: {datetime.now().isoformat()}\n\n")
            for be in best_edits:
                lf.write("----\n")
                lf.write(f"Article ID: {be.get('article_id')}\n")
                lf.write(f"Original Q: {be.get('original_question')}\n")
                lf.write(f"Original A: {be.get('original_answer')}\n\n")
                lf.write(f"Best edit number: {be.get('best_edit_number')}\n")
                lf.write(f"Critic approved: {be.get('critic_approved')}\n")
                lf.write(f"Critic score: {be.get('critic_score')}\n")
                lf.write(f"Critic reason: {be.get('critic_reason')}\n\n")
                lf.write("All evaluations:\n")
                for ev in be.get('all_evaluations', []):
                    lf.write(f"  Edit #{ev.get('edit_number')}: approved={ev.get('approved')} score={ev.get('score')} drift={ev.get('drift_score')}\n")
                    lf.write(f"    Reason: {ev.get('reason')}\n")
                    raw = ev.get('raw_response', '')
                    if raw:
                        lf.write("    Raw response:\n")
                        lf.write(raw[:4000] + ("\n...[truncated]\n" if len(raw) > 4000 else "\n"))
                    lf.write("\n")
                lf.write("\n")
        
        print(f"\nResults for {lang.upper()}:")
        print(f"  QA pairs: {len(best_edits)}")
        print(f"  Approved: {approved_count} ({approved_count/len(best_edits)*100:.1f}%)")
        print(f"  Average score: {sum(scores)/len(scores):.2f}")
        print(f"  Total edits evaluated: {total_edits}")
        print(f"  Saved to: {output_file}")
        
        all_best_edits[lang] = best_edits
        
        summary_stats['per_language'][lang] = {
            'qa_pairs': len(best_edits),
            'approved': approved_count,
            'approval_rate': approved_count / len(best_edits),
            'average_score': sum(scores) / len(scores),
            'total_edits_evaluated': total_edits
        }
        summary_stats['total_qa_pairs'] += len(best_edits)
        summary_stats['total_edits_evaluated'] += total_edits
    
    summary_stats['languages_processed'] = len(edits_by_language)
    all_scores = [s['average_score'] for s in summary_stats['per_language'].values()]
    all_approvals = [s['approved'] for s in summary_stats['per_language'].values()]
    
    summary_stats['average_score'] = sum(all_scores) / len(all_scores) if all_scores else 0.0
    summary_stats['approved_rate'] = sum(all_approvals) / summary_stats['total_qa_pairs']
    
    summary_file = output_folder / "summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    print(f"Languages: {summary_stats['languages_processed']}")
    print(f"Total QA pairs: {summary_stats['total_qa_pairs']}")
    print(f"Total edits evaluated: {summary_stats['total_edits_evaluated']}")
    print(f"Overall approval rate: {summary_stats['approved_rate']*100:.1f}%")
    print(f"Average score: {summary_stats['average_score']:.2f}")
    print(f"\nResults saved to: {output_folder}")
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    main()
