
import json
from pathlib import Path
from typing import List, Dict, Optional


# Dataset-specific language mappings
DATASET_LANGUAGES = {
    'tydiqa': ['en', 'fi', 'id', 'ru', 'sw', 'te', 'ar', 'bn', 'ko'],
    'xquad': ['en', 'ar', 'de', 'el', 'es', 'hi', 'ro', 'ru', 'th', 'tr', 'vi', 'zh'],
    'mgsm': ['en', 'ru', 'sw', 'te', 'bn', 'zh', 'ja', 'th', 'es', 'fr', 'de'],
    'mlqa': ['en', 'ar', 'de', 'es', 'hi', 'vi', 'zh']
}


def get_data_path(dataset: str = "tydiqa") -> Path:
    """
    Get path to dataset directory.
    
    Args:
        dataset: Dataset name ('tydiqa', 'xquad', 'mgsm', etc.)
    
    Returns:
        Path to dataset directory
    """
    current_dir = Path(__file__).parent
    
    # Build possible paths for the dataset
    possible_paths = [
        current_dir.parent / "data" / dataset,
        current_dir.parent.parent / "data" / dataset,
        current_dir.parent / "data " / dataset,  # Handle space in "data " folder
        current_dir.parent.parent / "data " / dataset,
        current_dir.parent.parent.parent / "data" / dataset,
        current_dir.parent.parent.parent / "data " / dataset,
    ]
    
    # For xquad, also check for xquad-master (GitHub clone name)
    if dataset == "xquad":
        xquad_master_paths = [
            current_dir.parent / "data" / "xquad-master",
            current_dir.parent.parent / "data" / "xquad-master",
            current_dir.parent.parent.parent / "data" / "xquad-master",
        ]
        possible_paths.extend(xquad_master_paths)
    
    # For mlqa, check both dev/ and test/ subdirectories
    if dataset == "mlqa":
        mlqa_paths = [
            current_dir.parent / "data" / "MLQA_V1",
            current_dir.parent.parent / "data" / "MLQA_V1",
            current_dir.parent.parent.parent / "data" / "MLQA_V1",
        ]
        possible_paths.extend(mlqa_paths)
    
    for data_path in possible_paths:
        if data_path.exists():
            return data_path
    
    raise FileNotFoundError(
        f"{dataset.upper()} data not found. Tried:\n" + "\n".join(str(p) for p in possible_paths) +
        f"\n\nPlease download {dataset.upper()} dataset to data/{dataset}/"
    )


def get_available_languages(dataset: str = "tydiqa") -> List[str]:
    """
    Get list of available languages for a dataset.
    
    Args:
        dataset: Dataset name ('tydiqa', 'xquad', 'mgsm')
    
    Returns:
        List of language codes
    """
    return DATASET_LANGUAGES.get(dataset, [])



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
    data_path = get_data_path("tydiqa")
    file_path = data_path / f"tydiqa-goldp-v1.1-{split}.json"
    
    if not file_path.exists():
        available_files = list(data_path.glob("*.json")) if data_path.exists() else []
        raise FileNotFoundError(
            f"File not found: {file_path}\n"
            f"Data directory: {data_path}\n"
            f"Available files: {[f.name for f in available_files]}"
        )
    
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


def load_xquad_by_language(
    language: str,
    max_samples: Optional[int] = None
) -> List[Dict]:
    """
    Load XQuAD data for a specific language.
    
    Args:
        language: Language code (e.g., 'en', 'ar', 'zh')
        max_samples: Maximum number of articles to load (None = all)
    
    Returns:
        List of articles with context and QA pairs (unified format)
    """
    data_path = get_data_path("xquad")
    file_path = data_path / f"xquad.{language}.json"
    
    if not file_path.exists():
        available_files = list(data_path.glob("*.json")) if data_path.exists() else []
        raise FileNotFoundError(
            f"File not found: {file_path}\n"
            f"Data directory: {data_path}\n"
            f"Available files: {[f.name for f in available_files]}"
        )
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    articles = []
    
    # Parse XQuAD format (similar to SQuAD)
    for article_data in data.get('data', []):
        for paragraph in article_data.get('paragraphs', []):
            context = paragraph.get('context', '')
            
            qa_pairs = []
            for qa in paragraph.get('qas', []):
                question = qa.get('question', '')
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

def load_mlqa_by_language(
    language: str,
    max_samples: Optional[int] = None,
    split: str = "dev"
) -> List[Dict]:
    """
    Load MLQA data for a specific language.
    MLQA is cross-lingual QA, we load monolingual pairs (same language for context and question).
    
    Args:
        language: Language code (e.g., 'en', 'ar', 'de', 'es', 'hi', 'vi', 'zh')
        max_samples: Maximum number of articles to load (None = all)
        split: Dataset split ('dev' or 'test')
    
    Returns:
        List of articles with context and QA pairs (unified format)
    """
    data_path = get_data_path("mlqa")
    
    # MLQA format: {split}-context-{lang}-question-{lang}.json (monolingual pairs)
    file_path = data_path / split / f"{split}-context-{language}-question-{language}.json"
    
    if not file_path.exists():
        available_files = list((data_path / split).glob("*.json")) if (data_path / split).exists() else []
        raise FileNotFoundError(
            f"File not found: {file_path}\n"
            f"Data directory: {data_path / split}\n"
            f"Available files: {[f.name for f in available_files]}\n"
            f"Expected format: {split}-context-{language}-question-{language}.json"
        )
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    articles = []
    
    # Parse MLQA format (similar to SQuAD)
    for article_data in data.get('data', []):
        for paragraph in article_data.get('paragraphs', []):
            context = paragraph.get('context', '')
            
            qa_pairs = []
            for qa in paragraph.get('qas', []):
                question = qa.get('question', '')
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


def load_dataset_by_language(
    dataset: str,
    language: str,
    max_samples: Optional[int] = None,
    split: str = "train"
) -> List[Dict]:
    """
    Universal loader: Load any supported dataset for a specific language.
    
    Args:
        dataset: Dataset name ('tydiqa', 'xquad', 'mgsm', 'mlqa')
        language: Language code
        max_samples: Maximum number of samples to load
        split: Dataset split ('train', 'dev', 'test')
    
    Returns:
        List of articles/problems in unified format:
        {
            'context': <passage/problem>,
            'qa_pairs': [{'question': ..., 'answer': ..., 'id': ...}],
            'language': <lang_code>
        }
    """
    if dataset == "tydiqa":
        return load_tydiqa_by_language(language, max_samples, split)
    elif dataset == "xquad":
        # XQuAD only has one split
        return load_xquad_by_language(language, max_samples)
    elif dataset == "mgsm":
        return load_mgsm_by_language(language, max_samples, split)
    elif dataset == "mlqa":
        return load_mlqa_by_language(language, max_samples, split)
    else:
        raise ValueError(
            f"Unsupported dataset: {dataset}\n"
            f"Supported: {list(DATASET_LANGUAGES.keys())}"
        )
