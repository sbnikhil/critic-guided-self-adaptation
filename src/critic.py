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

    def _get_format_criteria(self, format_type: str) -> str:
        """Get format-specific evaluation criteria"""
        criteria = {
            "rewrite": """
            - Does it preserve ALL information from the original passage?
            - Is the wording genuinely different (not just minor word changes)?
            - Are there NO hallucinated facts or external information added?
            - Is it fluent and natural in the target language?
            """,
            
            "chain_of_thought": """
            - Does each step follow logically from the passage?
            - Are all reasoning steps grounded in the passage content?
            - Is there NO external information or assumptions added?
            - Does it demonstrate structured, step-by-step thinking?
            """,
            
            "implications": """
            - Are all implications directly stated or clearly derivable from the passage?
            - Is there NO speculation or external knowledge added?
            - Are the implications factually accurate based on the passage?
            - Is it formatted as a clear, numbered list?
            """,
            
            "self_qa": """
            - Are ALL questions answerable from the passage alone?
            - Do the answers accurately reflect information in the passage?
            - Are there NO questions requiring external knowledge?
            - Is it formatted clearly as Q&A pairs?
            """
        }
        return criteria.get(format_type, "- Is it accurate, fluent, and relevant to the passage?")

    def evaluate_language_batch(self, qa_groups: list, min_score_threshold: float = 6.0) -> list:
        prompt_parts = []
        prompt_parts.append(
            "You will be given multiple question-answer pairs with several generated edits each.\n"
            "For each QA pair, choose the single BEST edit based on format-specific criteria.\n\n"
            "CRITICAL RULES:\n"
            "1. REJECT edits that add information not in the original passage (hallucinations)\n"
            "2. REJECT edits with scores below 6.0 (mark as rejected)\n"
            "3. Judge each edit based on its FORMAT TYPE (see criteria below)\n"
            "4. Prefer edits that are factually grounded in the passage\n"
        )

        for idx, qa in enumerate(qa_groups, 1):
            ctx = qa.get('context', '')[:800]
            q = qa.get('original_question', '')
            a = qa.get('original_answer', '')
            prompt_parts.append(f"\n=== QA {idx} ===\nContext: {ctx}\nQ: {q}\nA: {a}\nEdits:")

            for j, edit in enumerate(qa.get('edits', []), 1):
                fmt = edit.get('format_type', 'unknown')
                content = edit.get('generated_text', '')[:800]
                is_valid = edit.get('is_valid', True)
                validation_msg = edit.get('validation_message', '')
                
                # Add format-specific criteria
                criteria = self._get_format_criteria(fmt)
                
                status = "" if is_valid else f" [PRE-FILTERED: {validation_msg}]"
                prompt_parts.append(f"\n-- Edit {j} ({fmt}){status} --\n{content}")
                prompt_parts.append(f"Criteria for {fmt}:{criteria}\n")

        prompt_parts.append(
            f"\nFor EACH QA (in same order), output JSON with:\n"
            f'{{"article_id": <id>, "selected_index": <1-based index or -1 if all poor>, "score": <0-10>, "reason": <brief reason>}}\n'
            f"If ALL edits score below {min_score_threshold}, set selected_index to -1.\n"
            f"Respond ONLY with valid JSON array."
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
                    score = float(entry.get('score', 0.0))
                    selected_idx = int(entry.get('selected_index', -1))
                    
                    # Apply quality threshold
                    if score < min_score_threshold and selected_idx > 0:
                        selected_idx = -1
                        entry['reason'] = f"Score {score} below threshold {min_score_threshold}. " + entry.get('reason', '')
                    
                    results.append({
                        'selected_index': selected_idx,
                        'score': score,
                        'reason': entry.get('reason', ''),
                        'raw_response': text,
                        'approved': selected_idx > 0 and score >= min_score_threshold
                    })
                else:
                    results.append({
                        'selected_index': -1,
                        'score': 0.0,
                        'reason': 'Missing entry from critic response',
                        'raw_response': text,
                        'approved': False
                    })

            return results

        except Exception as e:
            err = str(e)
            return [{
                'selected_index': -1,
                'score': 0.0,
                'reason': f'Evaluation error: {err}',
                'raw_response': err,
                'approved': False
            } for _ in qa_groups]
