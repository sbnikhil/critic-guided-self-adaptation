from typing import Dict, List
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from src.constants import LANGUAGE_NAMES, EMBEDDING_MODEL
except ImportError:
    from constants import LANGUAGE_NAMES, EMBEDDING_MODEL


class MultiFormatSelfEditGenerator:

    GENERATION_FORMATS = {
        "implications": "List of implications derived from the passage",
        "rewrite": "Complete rewrite in different wording",
        "self_qa": "New question-answer pairs from the passage",
        "chain_of_thought": "Think step-by-step then list implications"
    }

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = None
    ):
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
    
    def _call_ollama(self, prompt: str, temperature: float = 0.7) -> str:
        try:
            messages = [{"role": "user", "content": prompt}]
            
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True
            )
            
            return generated_text.strip()
        except Exception as e:
            print(f"Generation error: {e}")
            return ""
    
    def _generate_implications(self, context: str, language: str, length: str = "normal") -> str:
        
        lang_name = LANGUAGE_NAMES.get(language, language)
        
        if length == "long":
            instruction = "produce a long list of implications"
        elif length == "very_long":
            instruction = "produce a very long list of implications"
        else:
            instruction = "produce a list of implications"
        
        prompt = f"""Let's read the following passage and {instruction} derived directly or indirectly from the content.

Passage:
{context}

Write in {lang_name} only.

Implications:
"""
        
        return self._call_ollama(prompt, temperature=0.8)
    
    def _generate_chain_of_thought(self, context: str, language: str) -> str:
        
        lang_name = LANGUAGE_NAMES.get(language, language)
        
        prompt = f"""Let's read the following passage, think step by step, and then produce a list of implications derived directly or indirectly from the content.

Passage:
{context}

First generate a "Thought Process" and then "Implications". Write in {lang_name} only.

Thought Process:
"""
        
        return self._call_ollama(prompt, temperature=0.8)
    
    def _generate_rewrite(self, context: str, language: str) -> str:
        
        lang_name = LANGUAGE_NAMES.get(language, language)
        
        prompt = f"""Let's read the following passage and rewrite it in a few different ways, each one separated by a newline.

Passage:
{context}

Write in {lang_name} only.

Rewritten passages:
"""
        
        return self._call_ollama(prompt, temperature=0.9)
    
    def _generate_self_qa(self, context: str, language: str) -> str:
        
        lang_name = LANGUAGE_NAMES.get(language, language)
        
        prompt = f"""Let's read the following passage and rewrite it in a question-answer format.

Passage:
{context}

Write in {lang_name} only.

Question 1: """
        
        return self._call_ollama(prompt, temperature=0.8)
    
    def _measure_drift(self, original_text: str, generated_text: str) -> Dict[str, float]:
        
        if not generated_text or not original_text:
            return {
                "overall_drift": 0.0,
                "semantic_similarity": 1.0
            }
        
        embeddings = self.semantic_model.encode([original_text, generated_text])
        
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        
        drift = 1.0 - similarity
        
        return {
            "overall_drift": float(drift),
            "semantic_similarity": float(similarity)
        }
    
    def generate_edit(self, context: str, question: str, answer: str, language: str, format_type: str = "self_qa") -> Dict:
        
        if format_type not in self.GENERATION_FORMATS:
            raise ValueError(f"Invalid format: {format_type}. Choose from {list(self.GENERATION_FORMATS.keys())}")
        
        if format_type == "implications":
            generated = self._generate_implications(context, language, length="normal")
        elif format_type == "rewrite":
            generated = self._generate_rewrite(context, language)
        elif format_type == "self_qa":
            generated = self._generate_self_qa(context, language)
        elif format_type == "chain_of_thought":
            generated = self._generate_chain_of_thought(context, language)
        else:
            generated = ""
        
        drift_metrics = self._measure_drift(context, generated)
        
        return {
            "original_context": context,
            "original_question": question,
            "original_answer": answer,
            "generated_text": generated,
            "format_type": format_type,
            "language": language,
            "drift_score": drift_metrics["overall_drift"],
            "semantic_similarity": drift_metrics["semantic_similarity"],
            "approach": "multi_format"
        }
    
    def generate_all_formats(self, context: str, question: str, answer: str, language: str) -> List[Dict]:
        
        results = []
        for format_type in self.GENERATION_FORMATS.keys():
            result = self.generate_edit(context, question, answer, language, format_type)
            results.append(result)
        
        return results
