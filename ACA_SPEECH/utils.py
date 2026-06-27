'''
This file is for generally useful functions that do not belong to one certain class.
For example for common string operations or custom operations. 
'''

import sounddevice as sd
import numpy as np

def strip_role_prefixes(text):
    text = str(text or "")
    for prefix in ("Examiner:", "Examiner", "Feedback:", "Feedback", "Student:", "Student"):
        if text.startswith(prefix):
            return text[len(prefix):].lstrip()
    return text

# Clean strings
def clean_string(text):
    text = strip_role_prefixes(text)
    text = text.replace("<br>", "")
    text = text.replace("\\", "")
    return text

def normalize_text(text: str) -> str:
    return " ".join((text or "").lower().split()).strip()

def normalize_check_label(value) -> str:
    if value is None:
        return "NO"

    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""

    label = str(value).strip().upper()
    label = label.replace("EXAMINER:", "").replace("FEEDBACK:", "").strip()
    label = label.strip(" .,:;`'\"")

    if label in {"YES", "NO", "STOP"}:
        return label

    if "STOP" in label:
        return "STOP"
    if "YES" in label:
        return "YES"
    if "NO" in label:
        return "NO"

    return "NO"

def is_near_duplicate(new_norm: str, base_norm: str, max_extra_words: int = 3) -> bool:
    if not new_norm or not base_norm:
        return False
    if new_norm == base_norm:
        return True
    if new_norm.startswith(base_norm):
        extra = new_norm[len(base_norm):].strip()
        if extra and len(extra.split()) <= max_extra_words:
            return True
    return False

# Set a cap on the allowed number of words
def hard_cap_words(text, max_words=220):
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."

# Set a cap on the allowed number of characters
def hard_cap_chars(text, max_chars=1990):
    if not text:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 3] + "..."


def merge_transcripts(previous: str, current: str, max_overlap_words: int = 10) -> str:
    prev = " ".join((previous or "").split()).strip()
    curr = " ".join((current or "").split()).strip()

    if not prev:
        return curr
    if not curr:
        return prev

    prev_l = prev.lower()
    curr_l = curr.lower()

    if curr_l.startswith(prev_l):
        return curr
    if prev_l.startswith(curr_l):
        return prev

    prev_words = prev.split()
    curr_words = curr.split()
    max_k = min(max_overlap_words, len(prev_words), len(curr_words))

    for k in range(max_k, 0, -1):
        prev_tail = " ".join(prev_words[-k:]).lower()
        curr_head = " ".join(curr_words[:k]).lower()
        if prev_tail == curr_head:
            return (prev + " " + " ".join(curr_words[k:])).strip()

    return f"{prev} {curr}".strip()

def extract_content(response_json): # for Willma and Mistral
    try:
        return response_json["choices"][0]["message"]["content"]
    except Exception:
        return "[ERROR]"

def extract_responses_content(resp_json): # for GPT
    if not isinstance(resp_json, dict):
        return "[ERROR]"

    output = resp_json.get("output")
    if isinstance(output, list):
        texts = []
        for item in output:
            content = item.get("content", [])
            for c in content:
                if c.get("type") in ("output_text", "text"):
                    texts.append(c.get("text", ""))
        return "".join(texts).strip() or "[ERROR]"

    return "[ERROR]"

def play_pcm_stream(pcm_iter, sample_rate: int, channels: int = 1, dtype=np.int16, block_frames: int = 2048):
        with sd.RawOutputStream(
                samplerate=sample_rate,
                channels=channels,
                dtype=dtype,
                blocksize=block_frames,
        ) as stream:
            for chunk in pcm_iter:
                if not chunk:
                    continue
                stream.write(chunk)
