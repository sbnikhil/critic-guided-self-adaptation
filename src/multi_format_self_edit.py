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
    
    def _call_ollama(self, prompt: str, temperature: float = 0.7, **kwargs) -> str:
        try:
            messages = [{"role": "user", "content": prompt}]
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)
            # Only allow supported generation arguments
            allowed_args = [
                "max_new_tokens", "temperature", "top_p", "do_sample",
                "no_repeat_ngram_size", "repetition_penalty", "pad_token_id", "eos_token_id"
            ]
            gen_args = {
                "max_new_tokens": 512,
                "temperature": temperature,
                "do_sample": True,
                "top_p": 0.9,
                "pad_token_id": int(self.tokenizer.eos_token_id) if self.tokenizer.eos_token_id is not None else None
            }
            for k in allowed_args:
                if k in kwargs and kwargs[k] is not None:
                    gen_args[k] = int(kwargs[k]) if k.endswith("_token_id") else kwargs[k]
            with torch.no_grad():
                if "attention_mask" in inputs:
                    outputs = self.model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        **gen_args
                    )
                else:
                    outputs = self.model.generate(
                        input_ids=inputs["input_ids"],
                        **gen_args
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
    
    def _generate_self_qa(self, context: str, question: str, answer: str, language: str) -> str:
        import re
        lang_name = LANGUAGE_NAMES.get(language, language)
        # Tight prompt with rules and few-shot examples
        prompt = (
            "TASK: Paraphrase the question minimally, keeping the same intent and answer type.\n"
            "HARD RULES:\n"
            "- Keep the same wh-word and attribute (Who/When/Where/How many/etc.).\n"
            "- Do NOT change the topic or attribute (e.g., don’t switch 'When'→'What').\n"
            "- The original answer must still be correct for the new question.\n"
            "- Output EXACTLY:\n"
            "  Question: <one line>\n"
            "  Answer: <verbatim original answer>\n"
            "\n"
            "Good Example:\n"
            "Original: When was quantum field theory developed?\n"
            "Paraphrase: In which decade was quantum field theory developed?\n"
            "Answer: 1920s\n"
            "Bad Example:\n"
            "Original: When was quantum field theory developed?\n"
            "Paraphrase: What is quantum field theory?\n"
            "Answer: 1920s\n"
            "\n"
            f"Context: {context[:500]}\n"
            f"Original Question: {question}\n"
            f"Original Answer: {answer}\n"
            "\nQuestion:"
        )

        decoding_args = dict(
            temperature=0.3,
            top_p=0.9,
            max_new_tokens=48,
            no_repeat_ngram_size=3,
            repetition_penalty=1.1,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id
        )

        def same_wh(oq, nq):
            WH = ["who","when","where","why","how","which","what","how many","how much"]
            def head(q):
                q = q.lower().strip()
                if q.startswith("how many"): return "how many"
                if q.startswith("how much"): return "how much"
                for w in WH:
                    if q.startswith(w): return w
                if re.match(r"(in|during)\s+which\s+year|which decade|in what year", q): return "when"
                if re.match(r"which person|which author|name the", q): return "who"
                return None
            return head(oq) == head(nq)

        def extract_keywords(q):
            tokens = re.findall(r"\b\w+\b", q.lower())
            stopwords = set(["the","a","an","of","in","on","at","for","to","by","with","and","or","is","was","were","are","as","from","that","this","which","who","when","where","why","how","what"])
            return [t for t in tokens if t not in stopwords and (t.isdigit() or len(t) > 2)]

        def content_overlap_ok(oq, nq):
            KEYS = extract_keywords(oq)
            return sum(k in nq.lower() for k in KEYS) >= max(1, round(len(KEYS)*0.4))

        def answer_ok(oa, na):
            return oa.strip().lower() in na.strip().lower()

        def context_ok(nq, context):
            return any(w in context.lower() for w in extract_keywords(nq))

        def valid_edit(oq, oa, nq, na, context, sim):
            return (same_wh(oq, nq)
                    and sim >= 0.90
                    and content_overlap_ok(oq, nq)
                    and answer_ok(oa, na)
                    and context_ok(nq, context)
                    and len(nq.split()) <= 20
                    and not nq.lower().startswith(("what is", "who awards", "what are the components")))

        best_candidate = None
        for attempt in range(3):
            generated_text = self._call_ollama(prompt, **decoding_args)
            match = re.search(r"Question:\s*(.+)\s*Answer:\s*(.+)", generated_text, re.DOTALL)
            if match:
                new_q = match.group(1).strip()
                new_a = match.group(2).strip()
            else:
                new_q = generated_text.strip().split("\n")[0]
                new_a = answer
            drift_metrics = self._measure_drift(question, new_q)
            drift_score = drift_metrics["overall_drift"]
            semantic_similarity = drift_metrics["semantic_similarity"]
            if valid_edit(question, answer, new_q, new_a, context, semantic_similarity):
                best_candidate = f"Question: {new_q}\nAnswer: {new_a}"
                break
        if not best_candidate:
            wh = same_wh(question, question) and question.split()[0] or "What"
            subj = " ".join(extract_keywords(question))
            fallback_q = f"{wh.capitalize()} {subj}?"
            best_candidate = f"Question: {fallback_q}\nAnswer: {answer}"
        return best_candidate
    
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
        if format_type != "self_qa":
            raise ValueError("Only self_qa format allowed for QA-only edits.")
        generated = self._generate_self_qa(context, question, answer, language)
        # Parse output
        import re
        match = re.search(r"Question:\s*(.+)\s*Answer:\s*(.+)", generated, re.DOTALL)
        if match:
            new_q = match.group(1).strip()
            new_a = match.group(2).strip()
        else:
            new_q = generated.strip().split("\n")[0]
            new_a = answer
        drift_metrics = self._measure_drift(question, new_q)
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