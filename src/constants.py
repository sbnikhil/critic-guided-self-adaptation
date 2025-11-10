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

# Default Ollama settings
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"

# Embedding model for drift measurement
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
