"""
Configuration-Based Self-Edit Generator

Generates self-edits by first creating generation configurations,
then applying them to produce paraphrased QA pairs.
"""

from typing import Dict, Tuple
import requests
import json
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from src.constants import LANGUAGE_NAMES, DEFAULT_OLLAMA_URL, DEFAULT_MODEL, EMBEDDING_MODEL
except ImportError:
    from constants import LANGUAGE_NAMES, DEFAULT_OLLAMA_URL, DEFAULT_MODEL, EMBEDDING_MODEL


class ConfigBasedSelfEditGenerator:
    """Generate self-edits using learned configurations"""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        ollama_url: str = DEFAULT_OLLAMA_URL
    ):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.semantic_model = SentenceTransformer(EMBEDDING_MODEL)
    
    def _call_ollama(self, prompt: str, temperature: float = 0.7) -> str:
        """Call Ollama API to generate text"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": temperature
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            print(f"Ollama API error: {e}")
            return ""
    
    def _generate_config(
        self,
        context: str,
        question: str,
        answer: str,
        language: str
    ) -> Dict:
        """Generate configuration for self-edit"""
        
        lang_name = LANGUAGE_NAMES.get(language, language)
        
        prompt = f"""Create a configuration to paraphrase this {lang_name} question-answer pair. The paraphrase must remain in {lang_name}.

Question: {question}
Answer: {answer}

Configuration Parameters:

1. semantic_preservation (float between 0.6-0.85):
   - 0.85: Very close to original meaning, minimal semantic changes
   - 0.75: Reasonably similar, moderate semantic flexibility
   - 0.65: More liberal paraphrasing while preserving core meaning
   - 0.60: Maximum paraphrasing freedom while staying semantically valid

2. lexical_diversity (float between 0.5-0.85):
   - 0.85: Replace many words with synonyms/alternatives
   - 0.70: Replace several words throughout
   - 0.60: Replace some words selectively
   - 0.50: Minimal word replacements

3. syntactic_variation (string):
   - "minimal": Keep sentence structure very similar
   - "moderate": Change some sentence structures
   - "high": Significantly restructure sentences

4. formality_level (string):
   - "same": Maintain the same level of formality
   - "more_formal": Make language more formal/polite
   - "less_formal": Make language more casual/conversational

Example output:
{{
  "semantic_preservation": 0.72,
  "lexical_diversity": 0.68,
  "syntactic_variation": "moderate",
  "formality_level": "same"
}}

Generate a diverse configuration (avoid always using the same values). Output ONLY valid JSON:"""

        response = self._call_ollama(prompt, temperature=0.9)
        
        try:
            config = json.loads(response)
            required_keys = ["semantic_preservation", "lexical_diversity", "syntactic_variation", "formality_level"]
            if all(key in config for key in required_keys):
                return config
        except:
            pass
        
        return {
            "semantic_preservation": 0.7,
            "lexical_diversity": 0.65,
            "syntactic_variation": "moderate",
            "formality_level": "same"
        }
    
    def _apply_config(
        self,
        context: str,
        question: str,
        answer: str,
        language: str,
        config: Dict
    ) -> Tuple[str, str]:
        """Apply configuration to generate paraphrased QA pair"""
        
        lang_name = LANGUAGE_NAMES.get(language, language)
        
        # Simple, direct prompt like SEAL's "rewrite" approach
        prompt = f"""Rewrite the following question-answer pair in {lang_name}. Make substantial changes while keeping the same meaning.

Original:
Q: {question}
A: {answer}

Rewrite this in a different way. Change the wording and sentence structure, but preserve the meaning. Output ONLY in {lang_name}.

Rewritten:
Q: """

        response = self._call_ollama(prompt, temperature=0.9)
        
        if not response:
            return question, answer
        
        return self._parse_response(response, question, answer)
    
    def _parse_response(
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
                # Handle both "QUESTION:" and "Q:" formats
                if line.startswith("QUESTION:"):
                    question = line.replace("QUESTION:", "").strip()
                elif line.startswith("Q:"):
                    question = line.replace("Q:", "").strip()
                elif line.startswith("ANSWER:"):
                    answer = line.replace("ANSWER:", "").strip()
                elif line.startswith("A:"):
                    answer = line.replace("A:", "").strip()
            
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
        original_text = f"{original_q} {original_a}"
        edited_text = f"{edited_q} {edited_a}"
        
        embeddings = self.semantic_model.encode([original_text, edited_text])
        
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        
        drift = 1.0 - similarity
        
        return {
            "overall_drift": float(drift),
            "semantic_similarity": float(similarity)
        }
    
    def generate_edit(
        self,
        context: str,
        question: str,
        answer: str,
        language: str
    ) -> Dict:
        """Generate self-edit using configuration-based approach"""
        
        config = self._generate_config(context, question, answer, language)
        
        edited_q, edited_a = self._apply_config(
            context, question, answer, language, config
        )
        
        drift_metrics = self._measure_drift(
            question, answer, edited_q, edited_a
        )
        
        return {
            "original_question": question,
            "original_answer": answer,
            "edited_question": edited_q,
            "edited_answer": edited_a,
            "language": language,
            "drift_score": drift_metrics["overall_drift"],
            "semantic_similarity": drift_metrics["semantic_similarity"],
            "generation_config": config,
            "approach": "config_based"
        }
