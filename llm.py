import requests
from typing import List
from config import Config
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class LLMHandler:
    def __init__(self):
        self.backend = Config.LLM_BACKEND
        if self.backend == "ollama":
            self.ollama_url = f"{Config.OLLAMA_URL}/api/generate"
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(Config.HF_MODEL)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(Config.HF_MODEL)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)

    def generate_answer(self, question: str, contexts: List[str]) -> str:
        context_text = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts)])
        if self.backend == "ollama":
            prompt = f"Context:\n{context_text}\nQuestion: {question}\nAnswer:"
            try:
                resp = requests.post(self.ollama_url, json={"model": Config.OLLAMA_MODEL, "prompt": prompt, "stream": False}, timeout=30)
                if resp.status_code == 200:
                    return resp.json().get('response', '').strip()
                return f"Error: {resp.status_code}"
            except Exception as e:
                return f"Error: {str(e)}"
        else:
            prompt = f"Context:\n{context_text}\nQuestion: {question}\nAnswer:"
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            outputs = self.model.generate(**inputs, max_length=Config.MAX_LENGTH, temperature=Config.TEMPERATURE)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
