"""Thin OpenAI wrapper. Agents take these functions as dependencies so tests can swap them."""
import json
import os
from functools import lru_cache

import faiss
import numpy as np
from openai import OpenAI

from . import config


class LLMUnavailable(RuntimeError):
    pass


@lru_cache
def _client() -> OpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise LLMUnavailable("OPENAI_API_KEY is not set. Add it to .env or your environment.")
    return OpenAI()


def embed(texts: list[str]) -> np.ndarray:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), 256):
        resp = _client().embeddings.create(model=config.EMBED_MODEL, input=texts[i : i + 256])
        vectors.extend(d.embedding for d in resp.data)
    arr = np.asarray(vectors, dtype="float32")
    faiss.normalize_L2(arr)
    return arr


def chat_json(system: str, user: str) -> dict:
    resp = _client().chat.completions.create(
        model=config.CHAT_MODEL,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return json.loads(resp.choices[0].message.content or "{}")
