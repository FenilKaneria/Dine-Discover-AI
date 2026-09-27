# Libraries to import to create our MCP server and handle data loading
from pathlib import Path
import json
import os
import sys

# FastMCP ships both as a standalone package and bundled inside the official
# `mcp` SDK. Prefer the standalone one, fall back to the bundled copy so the
# server runs even if `fastmcp` was never installed.
try:
    from fastmcp import FastMCP
except ImportError:  # pragma: no cover - depends on the local environment
    from mcp.server.fastmcp import FastMCP

# Allow `python src/server.py` from any working directory.
sys.path.insert(0, str(Path(__file__).parent))

from config import (  # noqa: E402
    CULINARY_MAP_PATH,
    RESTAURANT_DATA_PATH,
    REVIEW_DATA_PATH,
    ROOT_DIR,
)
from config import RERANK_MODEL, RERANKER  # noqa: E402
from retrieval import get_retriever  # noqa: E402
from build_index import parse_image_urls  # noqa: E402

# Initializing our MCP server instance
mcp = FastMCP("Dine-Discover-AI")

# Reranking of the top candidates (Jina cross-encoder by default, RERANKER=llm
# for gpt-oss-120b). Disable with RERANK=false.
RERANK = os.environ.get("RERANK", "true").lower() in ("1", "true", "yes")
RERANKER_LABEL = "LLM" if RERANKER == "llm" else RERANK_MODEL

# How many full records a single tool call may return. Without a cap, a broad
# query ("the") would dump all 210 records into the LLM context.
MAX_RESULTS = 5

# Loaded data is cached so we only touch the disk once per server process.
_CACHE: dict[str, object] = {}


# Helper functions
def _load_json(path: Path, cache_key: str) -> list[dict]:
    """Load a JSON list from disk once and cache it for the process lifetime."""
    if cache_key not in _CACHE:
        if not path.exists():
            raise FileNotFoundError(
                f"Required data file is missing: {path}. "
                "Run the project from the repository root so ./data is reachable."
            )
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list in {path}, got {type(data).__name__}.")
        _CACHE[cache_key] = data
    return _CACHE[cache_key]


def load_restaurant_data() -> list[dict]:
    """Load the structured restaurant dataset."""
    return _load_json(RESTAURANT_DATA_PATH, "restaurants")


def load_review_data() -> list[dict]:
    """Load the augmented user reviews."""
    return _load_json(REVIEW_DATA_PATH, "reviews")


def load_culinary_map() -> str:
    """Load the raw culinary map text (UTF-8, so accented names survive)."""
    if "culinary_map" not in _CACHE:
        if not CULINARY_MAP_PATH.exists():
            raise FileNotFoundError(f"Required data file is missing: {CULINARY_MAP_PATH}.")
        _CACHE["culinary_map"] = CULINARY_MAP_PATH.read_text(encoding="utf-8")
    return _CACHE["culinary_map"]


def summarize(restaurant: dict) -> dict:
    """Return a compact, LLM-friendly view of one restaurant record.

    Uses .get() throughout because a few records in the dataset are missing
    optional keys (e.g. "shortcomings")."""
    return {
        "name": restaurant.get("name", "Unknown"),
        "location": restaurant.get("location", "Unknown"),
        "type": restaurant.get("type", "Unknown"),
        "cuisine": restaurant.get("food_style", "Unknown"),
        "rating": restaurant.get("rating"),
        "price_range": "$" * int(restaurant["price_range"])
        if isinstance(restaurant.get("price_range"), (int, float))
        else "Unknown",
        "vibe": restaurant.get("vibe", "Unknown"),
        "environment": restaurant.get("environment", ""),
        "signature_dishes": restaurant.get("signatures", []),
        "shortcomings": restaurant.get("shortcomings", []),
    }


def error_payload(message: str) -> str:
    """Uniform error envelope so the agent never receives a raw traceback."""
    return json.dumps({"status": "error", "message": message}, indent=2)


# MCP Resource - Exposing the Raw Culinary Map data
@mcp.resource("culinary-map://california")
def get_culinary_map() -> str:
    """The full raw California Culinary Map text.
    Contains detailed descriptions of 100+ restaurants across California
    including their vibes, cuisines, ratings, and price ranges."""
    return load_culinary_map()


# TOOL 1 — Get Restaurant Info (Structured Search)
@mcp.tool()
def get_restaurant_info(restaurant_name: str) -> str:
    """Search for a restaurant by name and return its structured details
    including cuisine, rating, price range, signature dishes and vibe.
    Accepts partial names, e.g. "Iron" matches "Iron & Embers"."""
    try:
        restaurants = load_restaurant_data()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return error_payload(str(exc))

    query = (restaurant_name or "").lower().strip()
    if not query:
        return error_payload("Please provide a restaurant name to search for.")

    # Score matches so the closest name is ranked first: exact > prefix > substring.
    scored = []
    for restaurant in restaurants:
        name = str(restaurant.get("name", "")).lower()
        if not name:
            continue
        if name == query:
            score = 0
        elif name.startswith(query):
            score = 1
        elif query in name:
            score = 2
        # Reverse match handles the LLM passing a whole phrase instead of a name.
        # Guarded by length so short names like "Ivy" don't match everything.
        elif len(name) > 3 and name in query:
            score = 3
        else:
            continue
        scored.append((score, len(name), restaurant))

    if not scored:
        # No string match (typo, description instead of name): fall back to
        # semantic search over the restaurant embeddings.
        try:
            retriever = get_retriever()
            ids = retriever.resolve_name(restaurant_name, k=3)
        except Exception:
            ids = []
        if ids:
            return json.dumps(
                {
                    "status": "semantic_match",
                    "message": f"No exact name match for '{restaurant_name}'; closest restaurants by meaning:",
                    "results": [summarize(retriever.records[i]) for i in ids],
                },
                indent=2,
            )
        return json.dumps(
            {
                "status": "not_found",
                "message": f"No restaurant found matching '{restaurant_name}'.",
                "suggestion": "Try a partial name like 'Iron' or 'Sakura'.",
            },
            indent=2,
        )

    scored.sort(key=lambda item: (item[0], item[1]))
    results = [summarize(item[2]) for item in scored[:MAX_RESULTS]]

    return json.dumps(
        {
            "status": "found",
            "count": len(scored),
            "showing": len(results),
            "results": results,
        },
        indent=2,
    )


# TOOL 2 — Recommend by Vibe (hybrid RAG retrieval)
@mcp.tool()
def recommend_by_vibe(
    vibe: str,
    location: str | None = None,
    max_price: int | None = None,
    min_rating: float | None = None,
) -> str:
    """Recommend restaurants for a natural-language request, e.g. "romantic
    candlelit dinner", "cheap tacos near the beach", "zen sushi".
    Uses hybrid retrieval (Jina dense embeddings in ChromaDB + BM25, fused with
    Reciprocal Rank Fusion) and a reranker.
    Put the whole request (including words like "beach", "cheap", "rooftop")
    in `vibe` — semantic search handles them.
    Optional hard filters, only when the user states them explicitly:
    location = a named city or neighborhood (e.g. "Pasadena", "Little Tokyo"),
    never a generic word like "beach", "downtown" or "near me" (leave those in `vibe`);
    max_price = 1-4 ($ to $$$$): "$" or "cheapest" = 1, "under $$" or "$$ or less" = 2,
    "$$$ or less" = 3; words like "cheap" or "budget" alone are not a filter;
    min_rating = 0-5."""
    query = (vibe or "").strip()
    if not query:
        return error_payload("Please describe what you are looking for.")
    try:
        hits = get_retriever().search(
            query,
            k=MAX_RESULTS,
            location=location,
            max_price=max_price,
            min_rating=min_rating,
            rerank=RERANK,
        )
    except Exception as exc:
        return error_payload(f"Retrieval failed: {exc}")

    return json.dumps(
        {
            "query": query,
            "filters": {"location": location, "max_price": max_price, "min_rating": min_rating},
            "retrieval": "hybrid (dense + BM25, RRF)" + (f" + {RERANKER_LABEL} rerank" if RERANK else ""),
            "results": [
                {**summarize(h["record"]), "itemId": h["itemId"], "evidence": h["evidence"][:400]}
                for h in hits
            ],
        },
        indent=2,
    )


# TOOL 3 — Get Review (Joins restaurants to reviews via itemId)
@mcp.tool()
def get_review(restaurant_name: str) -> str:
    """Retrieve the full user review for a restaurant, including the reviewer's
    rating, review text and descriptions of the photos they uploaded."""
    try:
        restaurants = load_restaurant_data()
        reviews = load_review_data()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return error_payload(str(exc))

    query = (restaurant_name or "").lower().strip()
    if not query:
        return error_payload("Please provide a restaurant name to look up a review for.")

    # Reviews reference restaurants by itemId, not by name, so resolve the name
    # to its itemId(s) first and then join.
    # Rank candidates like get_restaurant_info (exact > prefix > substring) so
    # "Iron & Embers" never resolves to a longer name that merely contains it.
    reviews_by_item = {r.get("itemId"): r for r in reviews}
    candidates = []
    for restaurant in restaurants:
        name = str(restaurant.get("name", "")).lower()
        if not name or query not in name:
            continue
        review = reviews_by_item.get(restaurant.get("itemId"))
        if review is None:
            continue
        score = 0 if name == query else 1 if name.startswith(query) else 2
        candidates.append((score, len(name), restaurant, review))

    matching_review = None
    matched_name = None
    matched_location = None
    if candidates:
        candidates.sort(key=lambda item: (item[0], item[1]))
        _, _, restaurant, matching_review = candidates[0]
        matched_name = restaurant.get("name")
        matched_location = restaurant.get("location")

    # No name match: semantic search over the review texts.
    if not matching_review:
        try:
            retriever = get_retriever()
            hits = [h for h in retriever.search_passages(restaurant_name, k=8) if h["source"] == "review"]
        except Exception:
            hits = []
        if hits and hits[0]["similarity"] >= 0.45:
            matching_review = reviews_by_item.get(hits[0]["itemId"])
            record = retriever.records.get(hits[0]["itemId"], {})
            matched_name, matched_location = record.get("name"), record.get("location")

    # Return a not found message if no review matches the query
    if not matching_review:
        reviewed = [
            r.get("name")
            for r in restaurants
            if r.get("itemId") in reviews_by_item
        ]
        return json.dumps(
            {
                "status": "not_found",
                "message": f"No review found for '{restaurant_name}'.",
                "restaurants_with_reviews": reviewed,
            },
            indent=2,
        )

    return json.dumps(
        {
            "status": "found",
            "restaurant": matched_name,
            "location": matched_location,
            "reviewer": matching_review.get("userId", "Anonymous"),
            "rating": matching_review.get("rating"),
            "title": matching_review.get("title", ""),
            "review_text": matching_review.get("text", ""),
            "image_descriptions": matching_review.get("image_captions", []),
            "image_urls": parse_image_urls(matching_review.get("images")),
            "visit_date": matching_review.get("date", "N/A"),
        },
        indent=2,
    )


# TOOL 4 — Knowledge base search (RAG over prose + reviews)
@mcp.tool()
def search_knowledge_base(query: str) -> str:
    """Answer open questions ("which place has a greenhouse?", "where do
    reviewers mention lavender chicken?") by semantic search over the culinary
    map descriptions and user reviews. Returns the most relevant passages with
    their restaurant names and similarity scores — cite them in your answer."""
    if not (query or "").strip():
        return error_payload("Please provide a question to search for.")
    try:
        passages = get_retriever().search_passages(query, k=MAX_RESULTS)
    except Exception as exc:
        return error_payload(f"Retrieval failed: {exc}")
    return json.dumps({"query": query, "passages": passages}, indent=2)


# TOOL 5 — Multimodal image search (jina-clip-v2)
@mcp.tool()
def search_images(query: str, k: int = 3) -> str:
    """Find food / restaurant photos matching a text description, e.g.
    "margherita pizza with basil" or "sunlit dining room with plants".
    Uses jina-clip-v2, which embeds text and images in one vector space.
    Returns image paths/URLs; the chat UI renders them."""
    if not (query or "").strip():
        return error_payload("Please describe the image you are looking for.")
    try:
        images = get_retriever().search_images(query=query, k=max(1, min(int(k), 6)))
    except Exception as exc:
        return error_payload(f"Image search failed: {exc}")
    # CLIP similarities are only meaningful relative to each other: keep hits
    # close to the best one so an unrelated "3rd best" photo isn't shown.
    if images:
        best = images[0]["similarity"]
        images = [img for img in images if img["similarity"] >= 0.8 * best]
    for img in images:
        if img["source"] == "recipe":
            img["uri"] = str(ROOT_DIR / img["uri"])
    return json.dumps({"query": query, "images": images}, indent=2)


# Run the Server
if __name__ == "__main__":
    mcp.run()
