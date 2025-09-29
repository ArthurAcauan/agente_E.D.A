# agent/llm_client.py
import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_URL = os.environ.get("GEMINI_API_URL", "https://generativelanguage.googleapis.com")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Definir o endpoint correto para o modelo Gemini
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
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    out = r.json()
    try:
        parts = out["candidates"][0]["content"]["parts"]
        texts = [p.get("text", "") for p in parts if "text" in p]
        return "\n".join(texts).strip()
    except Exception:
        return json.dumps(out, indent=2)
