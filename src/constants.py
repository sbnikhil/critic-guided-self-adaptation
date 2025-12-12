"""
Shared Constants

Common constants used across the project.
"""

# Language codes to full names
LANGUAGE_NAMES = {
    'en': 'English',
    'ar': 'Arabic',
    'bn': 'Bengali',
    'fi': 'Finnish',
    'id': 'Indonesian',
    'ko': 'Korean',
    'ru': 'Russian',
    'sw': 'Swahili',
    'te': 'Telugu'
}

# Supported languages (9 in TyDi QA)
SUPPORTED_LANGUAGES = list(LANGUAGE_NAMES.keys())

# Embedding model for drift measurement
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
