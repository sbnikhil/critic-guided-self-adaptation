"""
Self-Edit Generator

Generates paraphrased QA pairs in the same language (X → X approach)
for multilingual continual learning.
"""

from typing import Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import from same package
try:
    from src.constants import LANGUAGE_NAMES, EMBEDDING_MODEL
    from src.config import Config, get_default_config
except ImportError:
    from constants import LANGUAGE_NAMES, EMBEDDING_MODEL
    from config import Config, get_default_config


class SelfEditGenerator:
    """Generate self-edits in the same language as input"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = None
    ):
        """
        Initialize the self-edit generator.
        
        Args:
            model_name: Hugging Face model name (default: Qwen2.5-1.5B-Instruct)
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        print(f"Loading model: {model_name}...")
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        print("Model loaded successfully!")
        self.semantic_model = SentenceTransformer(EMBEDDING_MODEL)
    
    def _call_ollama(self, prompt: str) -> str:
        """Call model to generate text"""
        try:
            messages = [{"role": "user", "content": prompt}]
            # Use chat template if supported by the tokenizer, otherwise fallback
            try:
                if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None):
                    inputs = self.tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                    )
                    # Move tensors to the model device
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                else:
                    # Fallback: tokenize the raw prompt
                    inputs = self.tokenizer(
                        prompt,
                        truncation=True,
                        return_tensors="pt",
                    )
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            except Exception as te:
                # On any tokenizer templating error, fallback to basic tokenization
                print(f"Tokenizer templating error, falling back: {te}")
                inputs = self.tokenizer(prompt, truncation=True, return_tensors="pt")
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=getattr(self.tokenizer, "eos_token_id", None)
                )

            # Decode the generated portion. If input_ids exists, slice off the prompt tokens.
            if "input_ids" in inputs:
                start_idx = inputs["input_ids"].shape[-1]
                generated_text = self.tokenizer.decode(
                    outputs[0][start_idx:],
                    skip_special_tokens=True
                )
            else:
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            return generated_text.strip()
        except Exception as e:
            print(f"Generation error: {e}")
            return ""
    
    def _parse_llm_response(
        self,
        response: str,
        fallback_q: str,
        fallback_a: str
    ) -> Tuple[str, str]:
        """Parse LLM response to extract question and answer"""
        try:
            lines = response.strip().split('\n')
            question = ""
            answer = ""
            
            for line in lines:
                line = line.strip()
                if line.startswith("QUESTION:"):
                    question = line.replace("QUESTION:", "").strip()
                elif line.startswith("ANSWER:"):
                    answer = line.replace("ANSWER:", "").strip()
            
            if not question or not answer:
                return fallback_q, fallback_a
            
            return question, answer
        except:
            return fallback_q, fallback_a
    
    def _measure_drift(
        self,
        original_q: str,
        original_a: str,
        edited_q: str,
        edited_a: str
    ) -> Dict[str, float]:
        """Measure semantic drift between original and edited versions"""
        # Combine question and answer for holistic similarity
        original_text = f"{original_q} {original_a}"
        edited_text = f"{edited_q} {edited_a}"
        
        # Get embeddings
        embeddings = self.semantic_model.encode([original_text, edited_text])
        
        # Calculate cosine similarity
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        
        # Drift is inverse of similarity
        drift = 1.0 - similarity
        
        return {
            "overall_drift": float(drift),
            "semantic_similarity": float(similarity)
        }
    
    def _high_preservation_edit(
        self,
        context: str,
        question: str,
        answer: str,
        language: str
    ) -> Tuple[str, str]:
        """Minimal changes - paraphrase while preserving structure"""
        
        lang_name = LANGUAGE_NAMES.get(language, language)
        
        prompt = f"""You are a multilingual question-answering expert.

CRITICAL: The original question and answer are in {lang_name}. You MUST keep your edited versions in {lang_name} as well. DO NOT translate to English or any other language.

Task: Create a SLIGHTLY DIFFERENT version of the question and answer that:
1. Maintains the same meaning and information
2. Uses different wording or phrasing
3. Stays in {lang_name} (the SAME LANGUAGE as the original - DO NOT TRANSLATE)
4. Makes minimal changes (conservative approach)

Context: {context[:500]}

Original Question ({lang_name}): {question}
Original Answer ({lang_name}): {answer}

Provide ONLY the edited question and answer in {lang_name} in this exact format:
QUESTION: <your edited question in {lang_name}>
ANSWER: <your edited answer in {lang_name}>"""

        response = self._call_ollama(prompt)
        
        if not response:
            return question, answer
        
        edited_q, edited_a = self._parse_llm_response(response, question, answer)
        return edited_q, edited_a
    
    def _medium_preservation_edit(
        self,
        context: str,
        question: str,
        answer: str,
        language: str
    ) -> Tuple[str, str]:
        """Balanced reformulation (medium preservation)"""
        
        lang_name = LANGUAGE_NAMES.get(language, language)

        prompt = f"""You are a multilingual question-answering expert.

CRITICAL: The original question and answer are in {lang_name}. You MUST keep your edited versions in {lang_name} as well. DO NOT translate to English or any other language.

Task: Create a MODERATELY DIFFERENT (MEDIUM-PRESERVATION) version that:
1. Maintains the same core meaning and information
2. Uses notably different wording, phrasing, or structure
3. Stays in {lang_name} (the SAME LANGUAGE as the original - DO NOT TRANSLATE)
4. Makes moderate changes while preserving semantics

Context: {context[:500]}

Original Question ({lang_name}): {question}
Original Answer ({lang_name}): {answer}

Provide ONLY the edited question and answer in {lang_name} in this exact format:
QUESTION: <your edited question in {lang_name}>
ANSWER: <your edited answer in {lang_name}>"""

        response = self._call_ollama(prompt)
        
        if not response:
            return question, answer
        
        edited_q, edited_a = self._parse_llm_response(response, question, answer)
        return edited_q, edited_a
    
    def _low_preservation_edit(
        self,
        context: str,
        question: str,
        answer: str,
        language: str
    ) -> Tuple[str, str]:
        """Significant reformulation for diversity (low preservation / high variance)"""
        
        lang_name = LANGUAGE_NAMES.get(language, language)

        prompt = f"""You are a multilingual question-answering expert.

CRITICAL: The original question and answer are in {lang_name}. You MUST keep your edited versions in {lang_name} as well. DO NOT translate to English or any other language.

Task: Create a SIGNIFICANTLY DIFFERENT (LOW-PRESERVATION / HIGH-VARIANCE) version that:
1. Maintains the same core meaning and information
2. Uses very different wording, structure, and phrasing
3. Stays in {lang_name} (the SAME LANGUAGE as the original - DO NOT TRANSLATE)
4. Makes substantial changes for maximum diversity while preserving semantics

Context: {context[:500]}

Original Question ({lang_name}): {question}
Original Answer ({lang_name}): {answer}

Provide ONLY the edited question and answer in {lang_name} in this exact format:
QUESTION: <your edited question in {lang_name}>
ANSWER: <your edited answer in {lang_name}>"""

        response = self._call_ollama(prompt)
        
        if not response:
            return question, answer
        
        edited_q, edited_a = self._parse_llm_response(response, question, answer)
        return edited_q, edited_a
    
    def generate_edit(
        self,
        context: str,
        question: str,
        answer: str,
        language: str,
        config: Config = None
    ) -> Dict:
        """
        Generate a self-edit for a QA pair.
        
        Args:
            context: Context passage
            question: Original question
            answer: Original answer
            language: Language code (e.g., 'en', 'ar')
            config: Configuration (uses default if None)
        
        Returns:
            Dictionary with original, edited, and drift metrics
        """
        if config is None:
            config = get_default_config()
        
        # Generate edit based on strategy
        strategy = config.formulation_strategy

        if strategy == "high_preservation":
            edited_q, edited_a = self._high_preservation_edit(
                context, question, answer, language
            )
        elif strategy == "medium_preservation":
            edited_q, edited_a = self._medium_preservation_edit(
                context, question, answer, language
            )
        elif strategy == "low_preservation":
            edited_q, edited_a = self._low_preservation_edit(
                context, question, answer, language
            )
        else:
            # Default to medium
            edited_q, edited_a = self._medium_preservation_edit(
                context, question, answer, language
            )
        
        # Measure drift
        drift_metrics = self._measure_drift(question, answer, edited_q, edited_a)
        
        return {
            "original_question": question,
            "original_answer": answer,
            "edited_question": edited_q,
            "edited_answer": edited_a,
            "language": language,
            "drift_score": drift_metrics["overall_drift"],
            "semantic_similarity": drift_metrics["semantic_similarity"],
            "config_strategy": config.formulation_strategy
        }
