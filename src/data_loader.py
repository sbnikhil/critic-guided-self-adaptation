"""
TyDi QA Data Loader

Loads TyDi QA dataset for multilingual question answering.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional


def get_data_path() -> Path:
    """Get path to TyDi QA data directory"""
    # Try multiple possible locations
    current_dir = Path(__file__).parent
    
    possible_paths = [
        current_dir.parent / "data" / "tydiqa",  # ../data/tydiqa from src/ (ONE level up)
        current_dir.parent.parent / "data" / "tydiqa",  # ../../data/tydiqa from src/ (TWO levels up)
        current_dir.parent.parent.parent / "data" / "tydiqa",  # Note: "data " with space
        Path("/Users/Patron/Desktop/Projects/NLP/Critic/data /tydiqa"),  # Absolute path
    ]
    
    for data_path in possible_paths:
        if data_path.exists():
            return data_path
    
    raise FileNotFoundError(
        f"TyDi QA data not found. Tried:\n" + "\n".join(str(p) for p in possible_paths) +
        "\n\nPlease download from: https://github.com/google-research-datasets/tydiqa"
    )


def get_available_languages() -> List[str]:
    """Get list of available languages in TyDi QA"""
    return ['en', 'ar', 'bn', 'fi', 'id', 'ko', 'ru', 'sw', 'te']


def load_tydiqa_by_language(
    language: str,
    max_samples: Optional[int] = None,
    split: str = "train"
) -> List[Dict]:
    """
    Load TyDi QA data for a specific language.
    
    Args:
        language: Language code (e.g., 'en', 'ar', 'bn')
        max_samples: Maximum number of articles to load (None = all)
        split: Dataset split ('train' or 'dev')
    
    Returns:
        List of articles with context and QA pairs
    """
    data_path = get_data_path()
    file_path = data_path / f"tydiqa-goldp-v1.1-{split}.json"
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    articles = []
    
    # Parse TyDi QA format - filter by language
    for article_data in data.get('data', []):
        for paragraph in article_data.get('paragraphs', []):
            context = paragraph.get('context', '')
            
            qa_pairs = []
            for qa in paragraph.get('qas', []):
                question = qa.get('question', '')
                qa_id = qa.get('id', '')
                
                # Language code is in ID - handle both formats:
                # "finnish--123456-0" or "finnish-123456-0"
                if '--' in qa_id:
                    qa_lang_full = qa_id.split('--')[0]
                elif '-' in qa_id:
                    qa_lang_full = qa_id.split('-')[0]
                else:
                    continue  # Skip if no language info
                
                # Map full name to code
                lang_map = {
                    'english': 'en', 'arabic': 'ar', 'bengali': 'bn',
                    'finnish': 'fi', 'indonesian': 'id', 'korean': 'ko',
                    'russian': 'ru', 'swahili': 'sw', 'telugu': 'te'
                }
                qa_lang = lang_map.get(qa_lang_full, qa_lang_full)
                
                # Filter by language
                if qa_lang != language:
                    continue
                
                # Get answer text
                answers = qa.get('answers', [])
                if answers:
                    answer = answers[0].get('text', '')
                else:
                    answer = ""
                
                if question and answer:
                    qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'id': qa.get('id', '')
                    })
            
            if qa_pairs:
                articles.append({
                    'context': context,
                    'qa_pairs': qa_pairs,
                    'language': language
                })
                
                if max_samples and len(articles) >= max_samples:
                    return articles
    
    return articles
