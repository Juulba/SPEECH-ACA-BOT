import httpx
from utils import extract_content
from setup.keys import WILLMA_API_KEY

WILLMA_URL = "https://willma.surf.nl/api/v0/chat/completions"

client = httpx.AsyncClient(
    headers={
        "X-API-KEY": WILLMA_API_KEY,
        "Content-Type": "application/json",
    },
    timeout=30.0
)

async def willma_call(user_prompt, system_prompt, model, temperature, max_tokens) -> str:

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": 1,
        "max_tokens": max_tokens,
    }

    try:
        response = await client.post(WILLMA_URL, json=payload)
        response.raise_for_status()
        return extract_content(response.json())
    except httpx.HTTPError as e:
        print("Willma API error:", e)
        return "[ERROR]"


async def get_simple_willma_call(prompt, model, temperature, max_tokens) -> str:

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": 1,
        "max_tokens": max_tokens,
    }

    try:
        response = await client.post(WILLMA_URL, json=payload)
        response.raise_for_status()
        return extract_content(response.json())
    except httpx.HTTPError as e:
        print("Willma API error:", e)
        return "[ERROR]"