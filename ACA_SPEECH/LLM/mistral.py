"""File for Mistral API calls"""
import os
import httpx
from utils import extract_content
from setup.keys import MISTRAL_API_KEY

# Disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Base Mistral endpoint
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

# HTTP client with headers
client = httpx.AsyncClient(
    headers={
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    },
    timeout=30.0
)

async def get_simple_mistral_call(prompt, model, temperature, max_tokens) -> str:
    try:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": 1,
            "max_tokens": max_tokens
        }
        response = await client.post(MISTRAL_API_URL, json=payload)
        response.raise_for_status()
        return extract_content(response.json())
    except httpx.HTTPError as e:
        print("Mistral API error:", e)
        return "[ERROR]"

async def mistral_call(user_prompt, system_prompt, model, temperature, max_tokens) -> str:
    try:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "top_p": 1,
            "max_tokens": max_tokens
        }
        response = await client.post(MISTRAL_API_URL, json=payload)
        response.raise_for_status()
        return extract_content(response.json())
    except httpx.HTTPError as e:
        print("Mistral API error:", e)
        return "[ERROR]"