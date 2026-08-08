# Libraries to import to create our MCP server and handle data loading
from pathlib import Path
import json

# FastMCP ships both as a standalone package and bundled inside the official
# `mcp` SDK. Prefer the standalone one, fall back to the bundled copy so the
# server runs even if `fastmcp` was never installed.
try:
    from fastmcp import FastMCP
except ImportError:  # pragma: no cover - depends on the local environment
    from mcp.server.fastmcp import FastMCP

# Initializing our MCP server instance
mcp = FastMCP("Connoisseur-Server")

# Data paths
DATA_DIR = Path(__file__).parent.parent / "data"
CULINARY_MAP_PATH = DATA_DIR / "raw" / "California-Culinary-Map.txt"
RESTAURANT_DATA_PATH = DATA_DIR / "processed" / "structured_restaurant_data.json"
REVIEW_DATA_PATH = DATA_DIR / "processed" / "augmented_user_review.json"

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
    """Load the structured restaurant data produced in Module 1."""
    return _load_json(RESTAURANT_DATA_PATH, "restaurants")


def load_review_data() -> list[dict]:
    """Load the augmented user reviews produced in Module 1."""
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
    """The full raw California Culinary Map text from Module 1.
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


# TOOL 2 — Recommend by Vibe (Semantic Search)
@mcp.tool()
def recommend_by_vibe(vibe: str) -> str:
    """Find restaurants that match a given vibe or atmosphere keyword.
    Searches the structured vibe/environment fields and the raw culinary map text.
    Examples of vibe keywords: "moody", "sun-drenched", "romantic", "zen"."""
    try:
        restaurants = load_restaurant_data()
        raw_text = load_culinary_map()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return error_payload(str(exc))

    vibe_lower = (vibe or "").lower().strip()
    if not vibe_lower:
        return error_payload("Please provide a vibe keyword to search for.")

    # Pass 1: search the structured fields that actually describe atmosphere.
    # (The dataset stores "vibe" as a free-text string, plus "environment",
    # "type" and "location" — there is no "vibes"/"cuisine"/"neighborhood" key.)
    structured_matches = []
    for restaurant in restaurants:
        haystack = " ".join(
            str(restaurant.get(key, ""))
            for key in ("vibe", "environment", "type", "food_style", "location")
        ).lower()
        if vibe_lower in haystack:
            structured_matches.append(summarize(restaurant))

    # Highest rated first so the agent leads with the strongest recommendation.
    structured_matches.sort(
        key=lambda r: r["rating"] if isinstance(r["rating"], (int, float)) else 0,
        reverse=True,
    )

    # Pass 2: search the raw text for extra colour the structured data lost.
    text_excerpts = []
    for para in raw_text.split("\n\n"):
        para = para.strip()
        if para and vibe_lower in para.lower():
            text_excerpts.append(para[:300])

    return json.dumps(
        {
            "vibe_searched": vibe,
            "match_count": len(structured_matches),
            "structured_matches": structured_matches[:MAX_RESULTS],
            "raw_text_excerpts": text_excerpts[:3],
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
    reviews_by_item = {r.get("itemId"): r for r in reviews}
    matching_review = None
    matched_name = None
    for restaurant in restaurants:
        name = str(restaurant.get("name", "")).lower()
        if not name or query not in name:
            continue
        review = reviews_by_item.get(restaurant.get("itemId"))
        if review is not None:
            matching_review = review
            matched_name = restaurant.get("name")
            break

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
            "reviewer": matching_review.get("userId", "Anonymous"),
            "rating": matching_review.get("rating"),
            "title": matching_review.get("title", ""),
            "review_text": matching_review.get("text", ""),
            "image_descriptions": matching_review.get("image_captions", []),
            "visit_date": matching_review.get("date", "N/A"),
        },
        indent=2,
    )


# Run the Server
if __name__ == "__main__":
    mcp.run()
