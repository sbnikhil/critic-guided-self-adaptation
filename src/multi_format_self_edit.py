
from typing import Dict, List, Optional
import numpy as np
import re
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from src.constants import LANGUAGE_NAMES, EMBEDDING_MODEL
except ImportError:
    from constants import LANGUAGE_NAMES, EMBEDDING_MODEL

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


class MultiFormatSelfEditGenerator:
    """
    Generate synthetic training data from passages (SEAL approach).
    
    Uses natural, conversational prompts with completion-style generation.
    
    Input: Passage/context
    Output: Transformed version in one of 4 formats
    
    No Q&A required - purely context-based transformation.
    """

    GENERATION_FORMATS = {
        "implications": "List of implications derived from passage",
        "rewrite": "Rewrite passage in different wording",
        "self_qa": "Question-answer pairs from passage",
        "chain_of_thought": "Step-by-step reasoning + implications"
    }


    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        load_model: bool = True
    ):
        """
        Initialize the generator.
        
        Args:
            model_name: HuggingFace model to use (default: Qwen2.5-7B-Instruct)
            device: 'cuda' or 'cpu' (auto-detected if None)
            load_model: Whether to load model (set False for testing)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        
        # Check available GPUs
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if self.num_gpus > 0:
            print(f"Found {self.num_gpus} GPU(s)")
        
        # Load semantic similarity model for drift measurement
        try:
            self.semantic_model = SentenceTransformer(EMBEDDING_MODEL)
            if self.device == "cuda":
                self.semantic_model = self.semantic_model.to(self.device)
        except Exception:
            self.semantic_model = None

        # Load generation model
        if load_model:
            try:
                print(f"Loading model: {self.model_name}")
                # Load tokenizer with fix for Mistral regex warning
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*fix_mistral_regex.*")
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                
                # Load model without device_map to avoid offloading issues with LoRA checkpoints
                if self.device == "cuda":
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name, 
                        torch_dtype=torch.float16,
                        device_map=None,  # Don't use auto device_map to avoid offloading
                    )
                    # Manually move to GPU
                    self.model = self.model.to(self.device)
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float32,
                    )
                
                print(f"Model loaded successfully on {self.device}")
                    
            except Exception as e:
                print(f"Warning: Failed to load {self.model_name}: {e}")
                print(f"Error details: {type(e).__name__}")
                self.tokenizer = None
                self.model = None


    def _generate_text(self, prompt: str, temperature: float = 0.7, max_new_tokens: int = 150) -> str:
        """Generate text from prompt using loaded model (completion-style)."""
        if self.model is None or self.tokenizer is None:
            return ""  # Model not loaded

        try:
            # Simple completion (no chat template - SEAL style)
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
            
            # Move to appropriate device
            if hasattr(self.model, 'device'):
                device = self.model.device
            else:
                # For models with device_map="auto", get the first device
                device = next(self.model.parameters()).device
            
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=(temperature > 0.0),
                    top_p=0.95,  # Nucleus sampling
                    top_k=50,    # Top-k sampling
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            # Decode only the generated part (skip input prompt)
            generated_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]:], 
                skip_special_tokens=True
            )
            
            return generated_text.strip()
            
        except Exception as e:
            print(f"Generation error: {e}")
            return ""
    
    def _generate_implications(self, context: str, language: str) -> str:
        """Generate list of implications from passage (SEAL-style)."""
        # Map language codes to names for clarity
        lang_names = {
            'te': 'Telugu', 'sw': 'Swahili', 'ru': 'Russian', 'fi': 'Finnish',
            'id': 'Indonesian', 'en': 'English', 'ar': 'Arabic', 'bn': 'Bengali',
            'ko': 'Korean', 'ja': 'Japanese', 'zh': 'Chinese', 'hi': 'Hindi',
            'es': 'Spanish', 'de': 'German', 'fr': 'French'
        }
        lang_name = lang_names.get(language, language)
        
        if language == 'en':
            # English passage - use English
            prompt = (
                f"Read the following passage and produce a list of implications "
                f"derived from the content.\n\n"
                f"Passage:\n{context}\n\n"
                f"Implications:\n"
            )
        else:
            # Non-English - be VERY explicit
            prompt = (
                f"TASK: Read this {lang_name} passage and write implications in {lang_name}.\n\n"
                f" CRITICAL RULES:\n"
                f"1. You MUST write ONLY in {lang_name}\n"
                f"2. DO NOT write in English\n"
                f"3. DO NOT translate - keep the same language as the passage\n"
                f"4. Every single word must be in {lang_name}\n\n"
                f"Passage ({lang_name}):\n{context}\n\n"
                f"Now write implications in {lang_name} (NOT English):\n"
            )
        return self._generate_text(prompt, temperature=0.7, max_new_tokens=450)

    def _generate_rewrite(self, context: str, language: str) -> str:
        """Rewrite passage in different wording (SEAL-style)."""
        lang_names = {
            'te': 'Telugu', 'sw': 'Swahili', 'ru': 'Russian', 'fi': 'Finnish',
            'id': 'Indonesian', 'en': 'English', 'ar': 'Arabic', 'bn': 'Bengali',
            'ko': 'Korean', 'ja': 'Japanese', 'zh': 'Chinese', 'hi': 'Hindi',
            'es': 'Spanish', 'de': 'German', 'fr': 'French'
        }
        lang_name = lang_names.get(language, language)
        
        context_tokens = len(context.split())
        max_tokens = min(max(context_tokens * 2, 300), 800)
        
        if language == 'en':
            prompt = (
                f"Read the following passage and rewrite it using different words "
                f"while keeping the same meaning.\n\n"
                f"Passage:\n{context}\n\n"
                f"Rewritten passage:\n"
            )
        else:
            prompt = (
                f"TASK: Rewrite this {lang_name} passage in {lang_name} using different words.\n\n"
                f" CRITICAL RULES:\n"
                f"1. You MUST write ONLY in {lang_name}\n"
                f"2. DO NOT write in English\n"
                f"3. Rewrite means: same meaning, different words, SAME LANGUAGE\n"
                f"4. Every single word must be in {lang_name}\n\n"
                f"Original passage ({lang_name}):\n{context}\n\n"
                f"Rewritten passage in {lang_name} (NOT English):\n"
            )
        return self._generate_text(prompt, temperature=0.7, max_new_tokens=max_tokens)

    def _generate_self_qa(self, context: str, language: str) -> str:
        """Generate Q&A pairs from passage (SEAL-style)."""
        lang_names = {
            'te': 'Telugu', 'sw': 'Swahili', 'ru': 'Russian', 'fi': 'Finnish',
            'id': 'Indonesian', 'en': 'English', 'ar': 'Arabic', 'bn': 'Bengali',
            'ko': 'Korean', 'ja': 'Japanese', 'zh': 'Chinese', 'hi': 'Hindi',
            'es': 'Spanish', 'de': 'German', 'fr': 'French'
        }
        lang_name = lang_names.get(language, language)
        
        if language == 'en':
            prompt = (
                f"Read the following passage and create question-answer pairs "
                f"based on the content.\n\n"
                f"Passage:\n{context}\n\n"
                f"Question-Answer Pairs:\n"
            )
        else:
            prompt = (
                f"TASK: Create question-answer pairs from this {lang_name} passage in {lang_name}.\n\n"
                f" CRITICAL RULES:\n"
                f"1. You MUST write ONLY in {lang_name}\n"
                f"2. DO NOT write questions or answers in English\n"
                f"3. Both questions AND answers must be in {lang_name}\n"
                f"4. Every single word must be in {lang_name}\n\n"
                f"Passage ({lang_name}):\n{context}\n\n"
                f"Question-Answer pairs in {lang_name} (NOT English):\n"
            )
        return self._generate_text(prompt, temperature=0.7, max_new_tokens=550)

    def _generate_chain_of_thought(self, context: str, language: str) -> str:
        """Generate step-by-step reasoning about passage (SEAL-style)."""
        lang_names = {
            'te': 'Telugu', 'sw': 'Swahili', 'ru': 'Russian', 'fi': 'Finnish',
            'id': 'Indonesian', 'en': 'English', 'ar': 'Arabic', 'bn': 'Bengali',
            'ko': 'Korean', 'ja': 'Japanese', 'zh': 'Chinese', 'hi': 'Hindi',
            'es': 'Spanish', 'de': 'German', 'fr': 'French'
        }
        lang_name = lang_names.get(language, language)
        
        if language == 'en':
            prompt = (
                f"Read the following passage, think step by step, and then produce "
                f"a list of implications.\n\n"
                f"Passage:\n{context}\n\n"
                f"Step-by-step analysis:\n"
            )
        else:
            prompt = (
                f"TASK: Analyze this {lang_name} passage step-by-step in {lang_name}.\n\n"
                f"CRITICAL RULES:\n"
                f"1. You MUST write ONLY in {lang_name}\n"
                f"2. DO NOT write your analysis in English\n"
                f"3. Think step-by-step but write each step in {lang_name}\n"
                f"4. Every single word of your reasoning must be in {lang_name}\n\n"
                f"Passage ({lang_name}):\n{context}\n\n"
                f"Step-by-step analysis in {lang_name} (NOT English):\n"
            )
        return self._generate_text(prompt, temperature=0.7, max_new_tokens=500)

    
    def _language_match_ratio(self, context: str, generated: str) -> float:
        """
        Approximate how well the generated text matches the language/script
        of the context.

        Strategy:
        1. For non-ASCII scripts (Telugu, Russian, Arabic, etc.):
           - Check character overlap
        2. For ASCII scripts (Finnish, Indonesian, etc.):
           - Check vocabulary overlap using word n-grams
           - Detect if generated is clearly English when context is not

        Returns:
            float: Ratio in [0.0, 1.0]. Higher = better language match.
        """
        # Check 1: Non-ASCII script matching
        ctx_chars = {ch for ch in context if ch.isalpha() and ord(ch) > 127}
        
        if ctx_chars:
            # Context has non-ASCII characters (Telugu, Russian, etc.)
            gen_alpha = [ch for ch in generated if ch.isalpha() and ord(ch) > 127]
            
            if not gen_alpha:
                # Context uses non-ASCII but generation doesn't → wrong script
                return 0.0
            
            # Compute character overlap
            match = sum(1 for ch in gen_alpha if ch in ctx_chars)
            return match / len(gen_alpha)
        
        # Check 2: ASCII language matching (Finnish, Indonesian, English, etc.)
        # Use vocabulary overlap and English detection
        
        # Extract words (lowercase, alphabetic only, length >= 3)
        def get_words(text):
            import re
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            return set(words)
        
        ctx_words = get_words(context)
        gen_words = get_words(generated)
        
        if not ctx_words or not gen_words:
            return 0.5  # Not enough signal
        
        # Check vocabulary overlap
        overlap = len(ctx_words & gen_words)
        overlap_ratio = overlap / len(gen_words) if gen_words else 0.0
        
        # Check if generated text is clearly English (common English words)
        common_english = {
            'the', 'and', 'was', 'that', 'with', 'from', 'this', 'were', 'have',
            'been', 'which', 'their', 'said', 'each', 'them', 'some', 'would',
            'these', 'into', 'than', 'also', 'his', 'her', 'had', 'are', 'but',
            'not', 'you', 'all', 'can', 'his', 'has', 'one', 'our', 'out', 'who',
            'when', 'what', 'where', 'will', 'more', 'other', 'they', 'about',
            'many', 'then', 'most', 'made', 'after', 'did', 'such', 'very',
            'through', 'between', 'without'
        }
        
        gen_english_count = sum(1 for w in gen_words if w in common_english)
        english_ratio = gen_english_count / len(gen_words) if gen_words else 0.0
        
        # Check if context has these English words too
        ctx_english_count = sum(1 for w in ctx_words if w in common_english)
        ctx_english_ratio = ctx_english_count / len(ctx_words) if ctx_words else 0.0
        
        # If context is low English but generation is high English → mismatch
        if ctx_english_ratio < 0.2 and english_ratio > 0.3:
            # Context is non-English but generation is mostly English
            return max(0.0, 1.0 - english_ratio)  # Penalize proportionally
        
        # If both have similar English ratios, use vocabulary overlap
        if abs(ctx_english_ratio - english_ratio) < 0.15:
            return max(overlap_ratio, 0.5)  # At least 0.5 if patterns match
        
        # Otherwise use overlap with some weight on English mismatch
        return max(0.0, min(1.0, overlap_ratio * (1.0 - abs(ctx_english_ratio - english_ratio))))
    
    def _measure_drift(self, original: str, generated: str) -> Dict[str, float]:
        """Measure semantic similarity between original and generated text."""
        if not generated or not original or self.semantic_model is None:
            return {"overall_drift": 0.0, "semantic_similarity": 1.0}
        
        embeddings = self.semantic_model.encode([original, generated])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        
        return {
            "overall_drift": float(1.0 - similarity),
            "semantic_similarity": float(similarity)
        }
    
    def _validate_edit(self, context: str, generated: str, format_type: str) -> tuple[bool, str, float]:
        """
        Validate generated edit quality.
        
        Checks:
        1. Minimum length
        2. Not cut-off
        3. Semantic similarity (0.4-0.98 range)
        4. Format-specific structure
        5. No hallucinated dates/facts
        6. No garbage from web scraping
        7. Language/script match
        
        Returns:
            tuple: (is_valid, validation_message, language_match_ratio)
        """
        # 1. Length check - increased minimum
        if len(generated.split()) < 20:
            return False, "Too short (< 20 words)", 0.0
        
        # 2. Cut-off check
        if generated.endswith(('...', 'dipl', 'impl', 'quest')):
            return False, "Text appears cut off", 0.0
        
        # 3. Garbage detection - catch Stack Exchange, homework sites, etc.
        garbage_markers = [
            '##', '###', '####',  # Markdown headers
            'Expert Answer', 'Question #', 'Hint:', 'Solution:',
            'This question has been deleted',
            'Not exactly what you\'re looking for',
            'Relevant Questions',
            '• @',  # User mentions (removed standalone '@' - too broad)
            'has been taken',
            'excerpt from a book',
            'Write a function',  # Code generation artifacts
            'def convert_to',  # Python code
            'Write an article based on this summary',  # Wrong task
            'You are a helpful assistant',  # Model meta-response
            'User\n\nAssistant:',  # Chat format leak
            'Here is an example of how you can'  # Meta-instructions
        ]
        
        gen_lower = generated.lower()
        for marker in garbage_markers:
            if marker.lower() in gen_lower:
                return False, f"Contains garbage marker: '{marker}'", 0.0
        
        # 3.5. Language-match check: ensure generated text mostly uses the same script
        # as the context. This helps catch cases where, for example, Telugu context
        # yields English output.
        lang_match = self._language_match_ratio(context, generated)
        if lang_match < 0.5:
            return False, f"Language/script mismatch (match_ratio={lang_match:.2f})", lang_match
        
        # 4. Similarity check
        metrics = self._measure_drift(context, generated)
        similarity = metrics["semantic_similarity"]
        
        if similarity < 0.4:
            return False, f"Low similarity: {similarity:.2f} (likely hallucination)", lang_match
        if similarity > 0.98:
            return False, f"Too similar: {similarity:.2f} (likely copy)", lang_match
        
        # 5. Format-specific checks
        if format_type == "self_qa" and not any(m in generated.lower() for m in ['q1:', 'q2:', '?', 'question']):
            return False, "No questions found", lang_match
        
        if format_type == "chain_of_thought":
            # Accept English numbers, "step" keyword, OR non-Latin numerals (Telugu, Arabic, etc.)
            has_structure = any(m in generated.lower() for m in ['step', '1.', '2.', '3.'])
            # Also check for various script numerals
            has_structure = has_structure or any(ch in generated for ch in [
                '౧', '౨', '౩',  # Telugu numerals
                '১', '২', '৩',  # Bengali numerals
                '۱', '۲', '۳',  # Arabic-Indic numerals
                '१', '२', '३',  # Devanagari numerals
                '1', '2', '3'   # ASCII numerals
            ])
            if not has_structure:
                return False, "No structured reasoning", lang_match
        
        if format_type == "implications" and not any(m in generated for m in ['1.', '2.', '-', '•']):
            return False, "No list structure", lang_match
        
        # 6. Hallucination check (dates)
        year_pattern = r'\b(?:19|20)\d{2}\b'
        context_years = set(re.findall(year_pattern, context))
        gen_years = set(re.findall(year_pattern, generated))
        new_years = gen_years - context_years
        
        if len(new_years) > 2:
            return False, f"Hallucinated dates: {new_years}", lang_match
        
        return True, "Valid", lang_match

    def generate_edit(self, context: str, language: str, format_type: str = "implications") -> Dict:
        """
        Generate synthetic data from context.
        
        Args:
            context: Passage to transform
            language: Language code ('en', 'ar', 'bn', etc.)
            format_type: One of ['implications', 'rewrite', 'self_qa', 'chain_of_thought']
        
        Returns:
            Dict with:
                - original_context: Input passage
                - generated_text: Synthetic transformation
                - format_type: Format used
                - language: Language code
                - drift_score: Semantic drift (0-1, lower = more similar)
                - semantic_similarity: Similarity (0-1, higher = more similar)
                - language_match_ratio: Script/character match (0-1, higher = better match)
                - is_valid: Whether edit passed quality checks
                - validation_message: Quality check details
        """
        if format_type not in self.GENERATION_FORMATS:
            raise ValueError(f"Invalid format: {format_type}. Choose from {list(self.GENERATION_FORMATS.keys())}")
        
        # Generate based on format type
        if format_type == "implications":
            generated = self._generate_implications(context, language)
        elif format_type == "rewrite":
            generated = self._generate_rewrite(context, language)
        elif format_type == "self_qa":
            generated = self._generate_self_qa(context, language)
        elif format_type == "chain_of_thought":
            generated = self._generate_chain_of_thought(context, language)
        else:
            generated = ""
        
        # Validate (now returns language_match_ratio as well)
        is_valid, validation_msg, lang_match_ratio = self._validate_edit(context, generated, format_type)
        
        # Measure drift
        metrics = self._measure_drift(context, generated)
        
        return {
            "original_context": context,
            "generated_text": generated,
            "format_type": format_type,
            "language": language,
            "drift_score": metrics["overall_drift"],
            "semantic_similarity": metrics["semantic_similarity"],
            "language_match_ratio": lang_match_ratio,
            "is_valid": is_valid,
            "validation_message": validation_msg
        }

    def generate_all_formats(self, context: str, language: str) -> List[Dict]:
        """
        Generate all 4 format types from a single context.
        
        Args:
            context: Passage to transform
            language: Language code
        
        Returns:
            List of 4 dicts (one per format)
        """
        results = []
        for format_type in self.GENERATION_FORMATS.keys():
            result = self.generate_edit(context, language, format_type)
            results.append(result)
        return results
    
    def generate_batch(self, contexts: List[str], language: str, format_type: str = "implications") -> List[Dict]:
        """
        Generate edits for a batch of contexts (faster with multi-GPU).
        
        Args:
            contexts: List of passages to transform
            language: Language code
            format_type: One of ['implications', 'rewrite', 'self_qa', 'chain_of_thought']
        
        Returns:
            List of dicts (one per context)
        """
        results = []
        for context in contexts:
            result = self.generate_edit(context, language, format_type)
            results.append(result)
        return results
