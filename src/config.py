"""Central configuration: environment, model names and data paths.

Every script imports from here so the LLM provider, embedding models and file
locations are defined exactly once.
"""
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# Unparseable .env lines are skipped silently; llm_config_problem() reports the
# consequences in plain English instead of leaking parser warnings.
logging.getLogger("dotenv.main").setLevel(logging.ERROR)

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = ROOT_DIR / "data"
CULINARY_MAP_PATH = DATA_DIR / "raw" / "California-Culinary-Map.txt"
RESTAURANT_DATA_PATH = DATA_DIR / "processed" / "structured_restaurant_data.json"
REVIEW_DATA_PATH = DATA_DIR / "processed" / "augmented_user_review.json"
RECIPE_DATA_PATH = DATA_DIR / "processed" / "augmented_food_recipe.json"
RECIPE_IMAGE_DIR = DATA_DIR / "raw" / "synthetic_recipe_images"
CHROMA_DIR = DATA_DIR / "chroma"
EVAL_DIR = ROOT_DIR / "eval"

# ---------------------------------------------------------------------------
# LLM (Groq, OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# GROQ_API_KEY is preferred; OPENAI_API_KEY is accepted for older .env files.
GROQ_API_KEY = (os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip()

# Fast model: tool routing, query generation. Strong model: reranking,
# grounded answers, LLM-as-judge.
CHAT_MODEL = os.environ.get("CHAT_MODEL", "openai/gpt-oss-20b")
REASONING_MODEL = os.environ.get("REASONING_MODEL", "openai/gpt-oss-120b")

# ---------------------------------------------------------------------------
# Embeddings (Jina AI API)
# ---------------------------------------------------------------------------
JINA_API_KEY = os.environ.get("JINA_API_KEY", "").strip()
JINA_EMBED_URL = "https://api.jina.ai/v1/embeddings"
EMBED_MODEL = os.environ.get("EMBED_MODEL", "jina-embeddings-v3")
IMAGE_EMBED_MODEL = os.environ.get("IMAGE_EMBED_MODEL", "jina-clip-v2")

# Reranking: "jina" = cross-encoder via the Jina API (fast, same key),
# "llm" = listwise rerank with REASONING_MODEL (slow, uses Groq quota).
JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
RERANK_MODEL = os.environ.get("RERANK_MODEL", "jina-reranker-v2-base-multilingual")
RERANKER = os.environ.get("RERANKER", "jina").strip().lower()

# Chroma collection names
RESTAURANTS_COLLECTION = "restaurants"
CULINARY_MAP_COLLECTION = "culinary_map"
REVIEWS_COLLECTION = "reviews"
IMAGES_COLLECTION = "images"


def llm_config_problem() -> str | None:
    """Return a human-readable problem with the API configuration, or None."""
    if not GROQ_API_KEY:
        return "GROQ_API_KEY is not set. Add it to the .env file in the project root."
    if not JINA_API_KEY:
        return "JINA_API_KEY is not set. Add it to the .env file in the project root."
    return None


def groq_client():
    """Raw OpenAI SDK client pointed at Groq."""
    from openai import OpenAI

    return OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL, max_retries=3, timeout=60)


def chat_model(model: str | None = None, temperature: float = 0.3, max_retries: int = 2):
    """LangChain chat model pointed at Groq."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model or CHAT_MODEL,
        api_key=GROQ_API_KEY,
        base_url=GROQ_BASE_URL,
        temperature=temperature,
        timeout=60,
        max_retries=max_retries,
    )


def answer_model():
    """The reasoning model for final answers, falling back to the fast model
    when it errors (e.g. Groq free-tier daily token limit on gpt-oss-120b)."""
    return chat_model(REASONING_MODEL, max_retries=1).with_fallbacks([chat_model(CHAT_MODEL)])
