"""
LLM Client with Grok, Phi-2, Gemini, OpenAI support - STRICT & STABLE VERSION
"""
import os
import time
from typing import Optional


class LLMClient:
    def __init__(
        self,
        model: str = "grok-4.3",
        provider: str = "grok",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 30,
    ):
        self.model = model
        self.provider = provider
        self.api_key = api_key
        self.max_retries = max_retries
        self.timeout = timeout
        self.client = None
        self._loaded = False

    # ─────────────────────────────────────────────
    # LOAD
    # ─────────────────────────────────────────────
    def load(self) -> bool:
        if self._loaded:
            return True

        try:
            # ───────────── GROK (NEW) ─────────────
            if self.provider == "grok":
                import requests
                api_key = self.api_key or os.environ.get("XAI_API_KEY")
                if not api_key:
                    print("[LLM] No Grok API key found. Set XAI_API_KEY or pass --llm-api-key")
                    print("[LLM] Get your key at: https://console.x.ai")
                    return False
                
                self.client = {
                    "base_url": "https://api.x.ai/v1",
                    "api_key": api_key
                }
                self._loaded = True
                print(f"[LLM] Grok {self.model} loaded ✓")
                return True

            # ───────────── GEMINI ─────────────
            elif self.provider == "gemini":
                import google.generativeai as genai
                
                api_key = self.api_key or os.environ.get("GEMINI_API_KEY")
                if not api_key:
                    print("[LLM] No Gemini API key found. Set GEMINI_API_KEY")
                    return False
                
                genai.configure(api_key=api_key)
                self.client = genai.GenerativeModel(self.model)
                self._loaded = True
                print(f"[LLM] Gemini {self.model} loaded ✓")
                return True

            # ───────────── OLLAMA ─────────────
            elif self.provider == "ollama":
                import requests
                self.client = {
                    "base_url": os.environ.get("OLLAMA_URL", "http://localhost:11434")
                }

                response = requests.get(f"{self.client['base_url']}/api/tags", timeout=5)
                if response.status_code == 200:
                    self._loaded = True
                    print(f"[LLM] Ollama ready ✓ (model: {self.model})")
                    return True
                else:
                    print("[LLM] Ollama not running. Start with: ollama serve")
                    return False

            # ───────────── TRANSFORMERS ─────────────
            elif self.provider == "transformers":
                try:
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                    import torch

                    print("[LLM] Loading Phi-2 (Transformers)...")

                    model_name = "microsoft/phi-2" if self.model == "phi:2.7b" else self.model

                    self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                    self.model_instance = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device_map="auto"
                    )

                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token

                    self._loaded = True
                    print("[LLM] Transformers model loaded ✓")
                    return True

                except ImportError:
                    print("[LLM] Install transformers + torch")
                    return False

            # ───────────── OPENAI ─────────────
            elif self.provider == "openai":
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key or os.environ.get("OPENAI_API_KEY"))
                self._loaded = True
                print("[LLM] OpenAI ready ✓")
                return True

            return False

        except Exception as e:
            print(f"[LLM] Load failed: {e}")
            return False

    # ─────────────────────────────────────────────
    # GENERATE (STRICT)
    # ─────────────────────────────────────────────
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 40,
        temperature: float = 0.0,  # deterministic
    ) -> Optional[str]:

        if not self._loaded:
            print("[LLM] Not loaded")
            return None

        for attempt in range(self.max_retries):
            try:

                # ───────────── GROK ─────────────
                if self.provider == "grok":
                    import requests
                    import json
                    
                    headers = {
                        "Authorization": f"Bearer {self.client['api_key']}",
                        "Content-Type": "application/json"
                    }
                    
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    messages.append({"role": "user", "content": prompt})
                    
                    payload = {
                        "model": self.model,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    }
                    
                    response = requests.post(
                        f"{self.client['base_url']}/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=self.timeout
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        text = data["choices"][0]["message"]["content"].strip()
                        return self._post_process(text)
                    else:
                        print(f"[LLM] Grok API error: {response.status_code} - {response.text}")
                        return None

                # ───────────── GEMINI ─────────────
                elif self.provider == "gemini":
                    full_prompt = prompt
                    if system_prompt:
                        full_prompt = f"{system_prompt}\n\n{prompt}"
                    
                    response = self.client.generate_content(
                        full_prompt,
                        generation_config={
                            "temperature": temperature,
                            "max_output_tokens": max_tokens,
                            "top_p": 0.9,
                        }
                    )
                    
                    if response and response.text:
                        text = response.text.strip()
                        return self._post_process(text)
                    return None

                # ───────────── OLLAMA ─────────────
                elif self.provider == "ollama":
                    import requests

                    response = requests.post(
                        f"{self.client['base_url']}/api/generate",
                        json={
                            "model": self.model,
                            "prompt": prompt,
                            "system": system_prompt,
                            "stream": False,
                            "options": {
                                "temperature": temperature,
                                "num_predict": max_tokens,
                                "top_p": 0.9,
                                "stop": ["\n", "OUTPUT:", "###"]
                            }
                        },
                        timeout=self.timeout
                    )

                    if response.status_code == 200:
                        text = response.json().get("response", "").strip()
                        return self._post_process(text)

                # ───────────── TRANSFORMERS ─────────────
                elif self.provider == "transformers":

                    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

                    inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model_instance.device)

                    outputs = self.model_instance.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=False,
                        temperature=0.0,
                        top_p=1.0,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

                    text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    text = text.replace(full_prompt, "").strip()

                    return self._post_process(text)

                # ───────────── OPENAI ─────────────
                elif self.provider == "openai":

                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt or ""},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )

                    text = response.choices[0].message.content.strip()
                    return self._post_process(text)

            except Exception as e:
                print(f"[LLM] Attempt {attempt+1} failed: {e}")
                time.sleep(1.5 * (attempt + 1))

        return None

    # ─────────────────────────────────────────────
    # POST PROCESS (VERY IMPORTANT)
    # ─────────────────────────────────────────────
    def _post_process(self, text: str) -> str:
        """
        Enforce single-line, clean output
        """
        if not text:
            return ""

        # Take first sentence only
        if "." in text:
            text = text.split(".")[0].strip() + "."

        # Remove newlines
        text = text.replace("\n", " ").strip()

        # Cleanup spacing
        while "  " in text:
            text = text.replace("  ", " ")

        return text

    # ─────────────────────────────────────────────
    def is_loaded(self) -> bool:
        return self._loaded