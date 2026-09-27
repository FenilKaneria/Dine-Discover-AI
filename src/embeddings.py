"""Jina AI embedding client (text + multimodal) used by the index and retriever.

- Text:  jina-embeddings-v3, asymmetric tasks — documents are embedded with
  "retrieval.passage" and user queries with "retrieval.query".
- Image: jina-clip-v2, which embeds text and images into one shared vector
  space, enabling text-to-image and image-to-image search.
"""
import base64
import io
import time
from pathlib import Path

import requests

from config import EMBED_MODEL, IMAGE_EMBED_MODEL, JINA_API_KEY, JINA_EMBED_URL, JINA_RERANK_URL, RERANK_MODEL

BATCH_SIZE = 64
IMAGE_BATCH_SIZE = 8  # images are large payloads; keep requests small
MAX_ATTEMPTS = 6


def _request(url: str, payload: dict) -> dict:
    """POST to a Jina API endpoint with retry on rate limits / server errors."""
    if not JINA_API_KEY:
        raise RuntimeError("JINA_API_KEY is not set in .env")
    headers = {"Authorization": f"Bearer {JINA_API_KEY}"}
    for attempt in range(MAX_ATTEMPTS):
        resp = None
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
        except requests.RequestException:
            if attempt == MAX_ATTEMPTS - 1:
                raise
        else:
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code not in (429, 500, 502, 503, 504) or attempt == MAX_ATTEMPTS - 1:
                raise RuntimeError(f"Jina API error {resp.status_code}: {resp.text[:300]}")
        # Token-per-minute limits reset on a 60 s window, so back off generously.
        time.sleep(20 * (attempt + 1) if resp is not None and resp.status_code == 429 else 2**attempt)
    raise RuntimeError("unreachable")


def _post(payload: dict) -> list[list[float]]:
    """Call the Jina embeddings API and return vectors in input order."""
    data = sorted(_request(JINA_EMBED_URL, payload)["data"], key=lambda d: d["index"])
    return [d["embedding"] for d in data]


def rerank(query: str, documents: list[str]) -> list[tuple[int, float]]:
    """Score documents against the query with the Jina cross-encoder reranker.
    Returns (document index, relevance score) pairs, best first."""
    if not documents:
        return []
    data = _request(
        JINA_RERANK_URL,
        {"model": RERANK_MODEL, "query": query, "documents": documents, "return_documents": False},
    )
    return [(int(r["index"]), float(r["relevance_score"])) for r in data["results"]]


def embed_texts(texts: list[str], task: str = "retrieval.passage") -> list[list[float]]:
    """Embed documents (task='retrieval.passage') or queries ('retrieval.query')."""
    vectors: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        vectors += _post(
            {
                "model": EMBED_MODEL,
                "task": task,
                "normalized": True,
                "input": texts[i : i + BATCH_SIZE],
            }
        )
    return vectors


def embed_query(text: str) -> list[float]:
    return embed_texts([text], task="retrieval.query")[0]


def image_to_base64(path: Path, max_side: int = 512) -> str:
    """Downscale an image and return it as base64 JPEG (keeps API payloads small)."""
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def embed_images(images: list[str]) -> list[list[float]]:
    """Embed images with jina-clip-v2. Each item is a URL or a base64 string."""
    vectors: list[list[float]] = []
    for i in range(0, len(images), IMAGE_BATCH_SIZE):
        batch = [{"image": img} for img in images[i : i + IMAGE_BATCH_SIZE]]
        vectors += _post({"model": IMAGE_EMBED_MODEL, "normalized": True, "input": batch})
    return vectors


def embed_clip_texts(texts: list[str]) -> list[list[float]]:
    """Embed text into the jina-clip-v2 space (for text-to-image search)."""
    vectors: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = [{"text": t} for t in texts[i : i + BATCH_SIZE]]
        vectors += _post({"model": IMAGE_EMBED_MODEL, "normalized": True, "input": batch})
    return vectors
