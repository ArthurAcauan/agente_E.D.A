# agent/llm_client.py
import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_URL = os.environ.get("GEMINI_API_URL", "https://generativelanguage.googleapis.com")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

model_name = "gemini-2.5-flash"
url = f"{GEMINI_API_URL}/v1beta/models/{model_name}:generateContent"

def call_gemini(prompt, max_tokens=2048, temperature=0.0):
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set in environment. See .env.example")
    headers = {
        "x-goog-api-key": GEMINI_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": temperature
        }
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60) 
    r.raise_for_status()
    out = r.json()

    if "candidates" in out and len(out["candidates"]) > 0:
        response_text = out["candidates"][0]["content"]["parts"][0]["text"]
    else:
        response_text = "Erro: Resposta inesperada do Gemini."

    return response_text

memory_text = ""  
user_q = ""  
quick_stats = {}  

prompt = f"Contexto: {memory_text}\nPergunta: {user_q}\nResumo: {quick_stats}"
max_tokens = 256  
