"""
File for OpenAI GPT API calls (Responses API)
"""
import os
import httpx
from utils import extract_responses_content
from setup.keys import OPENAI_API_KEY

os.environ["TOKENIZERS_PARALLELISM"] = "false"

OPENAI_API_URL = "https://api.openai.com/v1/responses"

client = httpx.AsyncClient(
    headers={
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    },
    timeout=30.0,
)


def _to_text(value) -> str:
    return "" if value is None else str(value)


def _msg(role: str, text: str) -> dict:
    return {
        "role": role,
        "content": [{"type": "input_text", "text": text}],
    }


def _normalize_max_output_tokens(value) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = 16
    return max(16, parsed)


def _log_openai_error(e: httpx.HTTPError):
    response = getattr(e, "response", None)
    if response is None:
        print("OpenAI API error:", e)
        return

    body = ""
    try:
        body = response.text
    except Exception:
        body = ""

    snippet = body[:800] if body else "<no body>"
    print(f"OpenAI API error: {e} | status={response.status_code} | body={snippet}")


async def get_simple_gpt_call(prompt, model, temperature, max_tokens) -> str:
    """
    Single-turn call (no system prompt).
    """
    prompt_text = _to_text(prompt).strip()
    if not prompt_text:
        return "[ERROR]"

    try:
        payload = {
            "model": model,
            "input": [_msg("user", prompt_text)],
            "temperature": temperature,
            "max_output_tokens": _normalize_max_output_tokens(max_tokens),
        }
        response = await client.post(OPENAI_API_URL, json=payload)
        response.raise_for_status()
        response_json = response.json()
        content = extract_responses_content(response_json)
        if isinstance(content, str) and content.startswith("[ERROR"):
            print(f"[OPENAI PARSE ERROR] get_simple_gpt_call model={model} returned {content}")
            if isinstance(response_json, dict):
                print(f"[OPENAI PARSE ERROR] response keys: {list(response_json.keys())}")
        return content
    except httpx.HTTPError as e:
        _log_openai_error(e)
        return "[ERROR]"


async def gpt_call(user_prompt, system_prompt, model, temperature, max_tokens) -> str:
    """
    System + user messages (chat-style) using the Responses API.
    """
    user_text = _to_text(user_prompt).strip()
    system_text = _to_text(system_prompt).strip()

    if not user_text:
        return "[ERROR]"

    try:
        input_messages = []
        if system_text:
            input_messages.append(_msg("system", system_text))
        input_messages.append(_msg("user", user_text))

        payload = {
            "model": model,
            "input": input_messages,
            "temperature": temperature,
            "max_output_tokens": _normalize_max_output_tokens(max_tokens),
        }
        response = await client.post(OPENAI_API_URL, json=payload)
        response.raise_for_status()
        response_json = response.json()
        content = extract_responses_content(response_json)
        if isinstance(content, str) and content.startswith("[ERROR"):
            print(f"[OPENAI PARSE ERROR] gpt_call model={model} returned {content}")
            if isinstance(response_json, dict):
                print(f"[OPENAI PARSE ERROR] response keys: {list(response_json.keys())}")
        return content
    except httpx.HTTPError as e:
        _log_openai_error(e)
        return "[ERROR]"