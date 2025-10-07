
# Python
"""
Embeddings + LLM generation helpers.
Uses OpenAI API via httpx. You can swap to another provider by changing
the two functions below: embed_texts() and chat_complete().
"""


import os
import httpx
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables from a .env file if presentu
load_dotenv()
import time

def pause_two_seconds():
    time.sleep(2)

OPENAI_BASE = os.environ.get("OPENAI_BASE", "https://api.openai.com/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")

# Safety check so errors are clear early.
if not OPENAI_API_KEY or OPENAI_API_KEY.strip() in {"", "OPENAI_API_KEY"}:
    raise RuntimeError(
        "OPENAI_API_KEY is            not set or invalid. Set it in your environment or .env file."
    )

# A single client for connection pooling.
_client = httpx.Client(
    base_url=OPENAI_BASE,
    headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
    timeout=60.0,
)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Takes a list of strings and returns a list of vectors (floats).
    We call the OpenAI embeddings endpoint in a single batch for efficiency.
    """
    # The OpenAI embeddings API accepts a list under "input".
    payload = {
        "model": EMBEDDING_MODEL,
        "input": texts,
    }

    r = _client.post("/embeddings", json=payload)
    r.raise_for_status()
    data = r.json()

    # The vectors are under data[i].embedding
    return [item["embedding"] for item in data["data"]]


def chat_complete(messages: List[Dict[str, str]]) -> str:
    """
    Calls a chat completion model with the given messages.
    messages format: [{"role": "system"/"user"/"assistant", "content": "..."}]
    """
    payload: Dict[str, Any] = {
        "model": CHAT_MODEL,
        "messages": messages,
        "temperature": 0.2,
    }
    r = _client.post("/chat/completions", json=payload)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]
