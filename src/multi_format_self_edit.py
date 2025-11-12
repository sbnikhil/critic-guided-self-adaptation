"""
Multi-Format Synthetic Data Generator (SEAL-Style)

Transforms passages into synthetic training data using 4 formats:
1. Implications - List of facts/inferences from passage
2. Rewrite - Paraphrase passage in different wording
3. Self-QA - Generate question-answer pairs from passage
4. Chain-of-thought - Step-by-step reasoning about passage

SEAL Approach: context → transformation (NO Q&A needed)
"""

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

# Use Qwen2.5-7B (same as SEAL)
DEFAULT_MODEL = "Qwen/Qwen2.5-7B"


class MultiFormatSelfEditGenerator:
    """
    Generate synthetic training data from passages (SEAL approach).
    
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
            model_name: HuggingFace model to use (default: Qwen2.5-7B)
            device: 'cuda' or 'cpu' (auto-detected if None)
            load_model: Whether to load model (set False for testing)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        
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
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, 
                    torch_dtype=(torch.float16 if self.device == "cuda" else torch.float32),
                    device_map="auto" if self.device == "cuda" else None
                )
                if self.device == "cpu":
                    self.model = self.model.to(self.device)
            except Exception as e:
                print(f"Warning: Failed to load {self.model_name}: {e}")
                self.tokenizer = None
                self.model = None


    def _generate_text(self, prompt: str, temperature: float = 0.7, max_new_tokens: int = 150) -> str:
        """Generate text from prompt using loaded model."""
        if self.model is None or self.tokenizer is None:
            return ""  # Model not loaded

        try:
            # Qwen uses ChatML format
            messages = [{"role": "user", "content": prompt}]
            
            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            inputs = self.tokenizer([text], return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate with better parameters for Qwen
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

            # Decode (skip input tokens)
            generated_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]:], 
                skip_special_tokens=True
            )
            
            # Clean up common artifacts
            generated_text = generated_text.strip()
            
            # Remove common Qwen artifacts
            if generated_text.startswith("assistant\n"):
                generated_text = generated_text[len("assistant\n"):].strip()
            
            return generated_text
            
        except Exception as e:
            print(f"Generation error: {e}")
            return ""


    # ========== Format-Specific Generators ==========
    
    def _generate_implications(self, context: str, language: str) -> str:
        """Generate list of implications from passage (SEAL format)."""
        lang_name = LANGUAGE_NAMES.get(language, language)
        
        prompt = f"""Extract factual implications from this passage. Only list facts that are directly stated or clearly implied. Do not add external information.

Passage: {context}

List 3-5 implications in this format:
1. [implication from passage]
2. [implication from passage]
3. [implication from passage]

Language: {lang_name}
Implications:"""
        return self._generate_text(prompt, temperature=0.3, max_new_tokens=300)

    def _generate_rewrite(self, context: str, language: str) -> str:
        """Rewrite passage in different wording (SEAL format)."""
        lang_name = LANGUAGE_NAMES.get(language, language)
        
        # Adaptive token limit based on input length
        context_tokens = len(context.split())
        max_tokens = min(max(context_tokens * 2, 200), 600)
        
        prompt = f"""Rewrite this passage using different words while keeping the exact same meaning. Do not add new information. Do not remove important details.

Original Passage: {context}

Rewritten Version (in {lang_name}):"""
        return self._generate_text(prompt, temperature=0.5, max_new_tokens=max_tokens)

    def _generate_self_qa(self, context: str, language: str) -> str:
        """Generate Q&A pairs from passage (SEAL format)."""
        lang_name = LANGUAGE_NAMES.get(language, language)
        
        prompt = f"""Read this passage and create 3-5 question-answer pairs. All questions MUST be answerable from the passage. Do not ask questions requiring outside knowledge.

Passage: {context}

Generate questions in this exact format:
Q1: [question]
A1: [answer from passage]
Q2: [question]
A2: [answer from passage]

Language: {lang_name}
Questions:"""
        return self._generate_text(prompt, temperature=0.5, max_new_tokens=400)

    def _generate_chain_of_thought(self, context: str, language: str) -> str:
        """Generate step-by-step reasoning about passage (SEAL format)."""
        lang_name = LANGUAGE_NAMES.get(language, language)
        
        prompt = f"""Analyze this passage step-by-step using ONLY information from the passage itself.

Passage: {context}

Provide your analysis in {lang_name} following this structure:

Step 1: [What is the main topic discussed in this passage?]
Step 2: [What are the key facts mentioned?]
Step 3: [What relationships or connections exist between these facts?]
Summary: [Summarize the passage in 1-2 sentences]

Your analysis:"""
        return self._generate_text(prompt, temperature=0.4, max_new_tokens=350)


    # ========== Quality Validation ==========
    
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
    
    def _validate_edit(self, context: str, generated: str, format_type: str) -> tuple[bool, str]:
        """
        Validate generated edit quality.
        
        Checks:
        1. Minimum length
        2. Not cut-off
        3. Semantic similarity (0.4-0.98 range)
        4. Format-specific structure
        5. No hallucinated dates/facts
        6. No garbage from web scraping
        """
        # 1. Length check - increased minimum
        if len(generated.split()) < 20:
            return False, "Too short (< 20 words)"
        
        # 2. Cut-off check
        if generated.endswith(('...', 'dipl', 'impl', 'quest')):
            return False, "Text appears cut off"
        
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
                return False, f"Contains garbage marker: '{marker}'"
        
        # 4. Similarity check
        metrics = self._measure_drift(context, generated)
        similarity = metrics["semantic_similarity"]
        
        if similarity < 0.4:
            return False, f"Low similarity: {similarity:.2f} (likely hallucination)"
        if similarity > 0.98:
            return False, f"Too similar: {similarity:.2f} (likely copy)"
        
        # 5. Format-specific checks
        if format_type == "self_qa" and not any(m in generated.lower() for m in ['q1:', 'q2:', '?', 'question']):
            return False, "No questions found"
        
        if format_type == "chain_of_thought" and not any(m in generated.lower() for m in ['step', '1.', '2.']):
            return False, "No structured reasoning"
        
        if format_type == "implications" and not any(m in generated for m in ['1.', '2.', '-', '•']):
            return False, "No list structure"
        
        # 6. Hallucination check (dates)
        context_years = set(re.findall(r'\b(19|20)\d{2}\b', context))
        gen_years = set(re.findall(r'\b(19|20)\d{2}\b', generated))
        new_years = gen_years - context_years
        
        if len(new_years) > 2:
            return False, f"Hallucinated dates: {new_years}"
        
        return True, "Valid"


    # ========== Public API ==========

    def generate_edit(self, context: str, language: str, format_type: str = "implications") -> Dict:
        """
        Generate synthetic data from context (SEAL-style).
        
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
        
        # Validate
        is_valid, validation_msg = self._validate_edit(context, generated, format_type)
        
        # Measure drift
        metrics = self._measure_drift(context, generated)
        
        return {
            "original_context": context,
            "generated_text": generated,
            "format_type": format_type,
            "language": language,
            "drift_score": metrics["overall_drift"],
            "semantic_similarity": metrics["semantic_similarity"],
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
