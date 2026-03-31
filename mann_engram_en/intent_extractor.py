import json
import re
import torch
from typing import Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI

class IntentExtractor:
    """Extracts core Query and Context using either BYOK Cloud API or an Ultra-Fast Local Model."""
    
    def __init__(self, device: str = "cpu", enable_local: bool = False, local_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.device = device
        self.enable_local = enable_local
        self.local_model = None
        self.local_tokenizer = None
        
        if self.enable_local:
            print(f"[IntentExtractor] Loading Hardcore Local Model ({local_model_id})...")
            try:
                self.local_tokenizer = AutoTokenizer.from_pretrained(local_model_id)
                self.local_model = AutoModelForCausalLM.from_pretrained(
                    local_model_id, 
                    torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
                    low_cpu_mem_usage=True
                ).to(self.device)
                self.local_model.eval()
                print("[IntentExtractor] Local Intent Engine warmed up successfully! (Zero Data Leakage)")
            except Exception as e:
                print(f"[Error] Failed to load local intent model: {e}. Falling back to Cloud/Bypass.")
                self.enable_local = False

    def _extract_with_local(self, raw_text: str) -> Dict[str, str]:
        """Uses the small local LLM and Regex to robustly extract intent."""
        prompt = f"""You are a precision AI parser. Extract the user's core question and useful background info from the messy input.
Output ONLY a JSON object with EXACTLY two keys: "query" and "context".

Raw Input: {raw_text}

JSON Output:"""
        
        messages = [
            {"role": "system", "content": "You are a helpful JSON data extractor."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            text_prompt = self.local_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.local_tokenizer([text_prompt], return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.local_model.generate(
                    **inputs, 
                    max_new_tokens=150,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.local_tokenizer.eos_token_id
                )
            
            response_text = self.local_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # Robust Regex fallback
            json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                return {"query": result.get("query", raw_text), "context": result.get("context", "")}
            else:
                return {"query": raw_text, "context": ""}
        except Exception as e:
            print(f"[Warning] Local intent extraction failed: {e}")
            return {"query": raw_text, "context": ""}

    def parse_user_input(self, raw_text: str, openai_client: Optional[OpenAI] = None, cloud_model: str = "gpt-4o-mini") -> Dict[str, str]:
        """Routes the parsing request to Cloud API or Local Engine based on configuration."""
        if len(raw_text.strip()) < 20:
            return {"query": raw_text, "context": ""}
            
        # Strategy A: BYOK Cloud Mode
        if openai_client:
            system_prompt = """You are a precision parser. Extract the user's core question and useful background info from the messy input. Ignore complaints. Output ONLY JSON with keys: "query" and "context"."""
            try:
                response = openai_client.chat.completions.create(
                    model=cloud_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Raw Input: {raw_text}"}
                    ],
                    response_format={ "type": "json_object" },
                    temperature=0.0
                )
                result = json.loads(response.choices[0].message.content)
                return {"query": result.get("query", raw_text), "context": result.get("context", "")}
            except Exception as e:
                print(f"[Warning] Cloud intent extraction failed: {e}. Trying local fallback...")

        # Strategy B: Hardcore Local Mode
        if self.enable_local and self.local_model:
            return self._extract_with_local(raw_text)
            
        # Strategy C: Bypass
        return {"query": raw_text, "context": ""}