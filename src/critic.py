"""
Gemini Critic
"""

import os
import google.generativeai as genai


class Critic:
    
    def __init__(self, model_name: str = "gemini-2.5-flash", temperature: float = 0.1, api_key: str = None):
        self.model_name = model_name
        self.temperature = temperature
        
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found")
        
        genai.configure(api_key=api_key)
        
        generation_config = {
            "temperature": self.temperature,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8000,
        }
        
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            safety_settings=safety_settings
        )

    def evaluate_language_batch(self, qa_groups: list) -> list:
        prompt_parts = []
        prompt_parts.append(
            "You will be given multiple question-answer pairs with several generated edits each.\n"
            "For each QA pair, choose the single BEST edit (the most useful, correct, and fluent)."
        )

        for idx, qa in enumerate(qa_groups, 1):
            ctx = qa.get('context', '')[:800]
            q = qa.get('original_question', '')
            a = qa.get('original_answer', '')
            prompt_parts.append(f"\n=== QA {idx} ===\nContext: {ctx}\nQ: {q}\nA: {a}\nEdits:")

            for j, edit in enumerate(qa.get('edits', []), 1):
                fmt = edit.get('format_type', 'unknown')
                content = edit.get('generated_text', '')[:800]
                prompt_parts.append(f"\n-- Edit {j} ({fmt}) --\n{content}\n")

        prompt_parts.append(
            "\nNow for EACH QA (in the same order), output a JSON array of objects with keys:\n"
            '{"article_id": <article_id>, "selected_index": <1-based edit index>, "score": <0-10>, "reason": <brief reason>}\n'
            "Respond ONLY with valid JSON (an array) and no extra text."
        )

        prompt = "\n".join(prompt_parts)

        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()

            import json, re
            m = re.search(r"(\[\s*\{.*\}\s*\])", text, re.S)
            json_text = None
            if m:
                json_text = m.group(1)
            else:
                start = text.find('[')
                end = text.rfind(']')
                if start != -1 and end != -1 and end > start:
                    json_text = text[start:end+1]

            if not json_text:
                raise ValueError('Could not parse JSON from critic response')

            parsed = json.loads(json_text)

            results = []
            for i, qa in enumerate(qa_groups):
                if i < len(parsed):
                    entry = parsed[i]
                    results.append({
                        'selected_index': int(entry.get('selected_index', -1)),
                        'score': float(entry.get('score', 0.0)),
                        'reason': entry.get('reason', ''),
                        'raw_response': text
                    })
                else:
                    results.append({
                        'selected_index': -1,
                        'score': 0.0,
                        'reason': 'Missing entry from critic response',
                        'raw_response': text
                    })

            return results

        except Exception as e:
            err = str(e)
            return [{
                'selected_index': -1,
                'score': 0.0,
                'reason': f'Evaluation error: {err}',
                'raw_response': err
            } for _ in qa_groups]
