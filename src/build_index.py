"""Build (or incrementally refresh) the ChromaDB vector index.

Collections (all cosine space, embeddings supplied by Jina):
  restaurants   one document per structured record            jina-embeddings-v3
  culinary_map  one chunk per culinary-map paragraph           jina-embeddings-v3
  reviews       one chunk per review (+ image captions)        jina-embeddings-v3
  images        recipe photos + review photos                  jina-clip-v2

Every chunk carries the restaurant's itemId so results from any collection can
be joined back to the structured record.

Idempotent: each chunk stores a content hash; unchanged chunks are not
re-embedded and chunks whose source disappeared are deleted.

Usage:
    python src/build_index.py            # incremental
    python src/build_index.py --rebuild  # drop and rebuild everything
    python src/build_index.py --skip-images
"""
import argparse
import ast
import hashlib
import json
import re
import sys

import chromadb

from config import (
    CHROMA_DIR,
    CULINARY_MAP_COLLECTION,
    CULINARY_MAP_PATH,
    IMAGES_COLLECTION,
    RECIPE_DATA_PATH,
    RECIPE_IMAGE_DIR,
    RESTAURANT_DATA_PATH,
    RESTAURANTS_COLLECTION,
    REVIEW_DATA_PATH,
    REVIEWS_COLLECTION,
)
from embeddings import embed_clip_texts, embed_images, embed_texts, image_to_base64

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_json(path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_map_paragraphs() -> list[str]:
    text = CULINARY_MAP_PATH.read_text(encoding="utf-8", errors="replace")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    # Drop the title / horizontal-rule header.
    return [p for p in paragraphs if not p.startswith("#") and p != "---"]


def strip_markdown(text: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"\1", text).strip()


def price_symbol(value) -> str:
    return "$" * int(value) if isinstance(value, (int, float)) else "unknown"


# ---------------------------------------------------------------------------
# Chunk builders — shared with restaurant_data_management.py for live sync
# ---------------------------------------------------------------------------


def restaurant_document(r: dict) -> str:
    """Natural-language document for one restaurant record."""
    signatures = ", ".join(r.get("signatures") or []) or "none listed"
    shortcomings = ", ".join(r.get("shortcomings") or []) or "none noted"
    return (
        f"{r.get('name', 'Unknown')} is a {r.get('type', 'restaurant')} in {r.get('location', 'California')} "
        f"serving {r.get('food_style', 'food')}. "
        f"Vibe: {r.get('vibe', '')}. Setting: {r.get('environment', '')}. "
        f"Signature dishes: {signatures}. Shortcomings: {shortcomings}. "
        f"Rating {r.get('rating', 'n/a')}/5, price {price_symbol(r.get('price_range'))}."
    )


def restaurant_metadata(r: dict) -> dict:
    """Flat metadata for Chroma `where` filters (Chroma only allows scalars)."""
    return {
        "itemId": int(r["itemId"]),
        "name": str(r.get("name", "")),
        "location": str(r.get("location", "")),
        "type": str(r.get("type", "")),
        "food_style": str(r.get("food_style", "")),
        "rating": float(r["rating"]) if isinstance(r.get("rating"), (int, float)) else 0.0,
        "price_range": int(r["price_range"]) if isinstance(r.get("price_range"), (int, float)) else 0,
    }


def restaurant_chunks(restaurants: list[dict]) -> list[dict]:
    return [
        {
            "id": f"rest-{r['itemId']}",
            "document": restaurant_document(r),
            "metadata": restaurant_metadata(r),
        }
        for r in restaurants
    ]


def map_chunks(restaurants: list[dict], paragraphs: list[str]) -> list[dict]:
    """Link each culinary-map paragraph to a restaurant.

    The map was written from the same records in the same order, so paragraph i
    describes record i; verify with a name check and fall back to a name search.
    """
    chunks = []
    for i, para in enumerate(paragraphs):
        clean = strip_markdown(para)
        match = None
        if i < len(restaurants) and restaurants[i].get("name", "")[:8] in para:
            match = restaurants[i]
        else:
            match = next((r for r in restaurants if r.get("name") and r["name"] in clean), None)
        chunks.append(
            {
                "id": f"map-{i}",
                "document": clean,
                "metadata": {
                    "itemId": int(match["itemId"]) if match else -1,
                    "name": match.get("name", "") if match else "",
                },
            }
        )
    return chunks


def parse_image_urls(value) -> list[str]:
    """Reviews store image URLs as a stringified Python list."""
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str) and value.strip():
        try:
            parsed = ast.literal_eval(value)
            return [str(v) for v in parsed] if isinstance(parsed, (list, tuple)) else [value]
        except (ValueError, SyntaxError):
            return [value]
    return []


def review_chunks(reviews: list[dict], restaurants: list[dict]) -> list[dict]:
    names = {r["itemId"]: r.get("name", "") for r in restaurants}
    chunks = []
    for rv in reviews:
        captions = " ".join(rv.get("image_captions") or [])
        document = f"Review of {names.get(rv.get('itemId'), 'a restaurant')}: {rv.get('title', '')}. {rv.get('text', '')}"
        if captions:
            document += f" Photos show: {captions}"
        chunks.append(
            {
                "id": f"review-{rv['reviewId']}",
                "document": document,
                "metadata": {
                    "itemId": int(rv.get("itemId", -1)),
                    "name": names.get(rv.get("itemId"), ""),
                    "rating": float(rv.get("rating") or 0),
                    "date": str(rv.get("date", "")),
                },
            }
        )
    return chunks


def image_chunks(reviews: list[dict], restaurants: list[dict]) -> list[dict]:
    """Recipe photos (local files) and review photos (remote URLs)."""
    chunks = []
    for recipe in load_json(RECIPE_DATA_PATH):
        path = RECIPE_IMAGE_DIR / f"recipe{recipe['id']}.png"
        if not path.exists():
            continue
        chunks.append(
            {
                "id": f"img-recipe-{recipe['id']}",
                "document": recipe.get("image_description") or recipe.get("name", ""),
                "metadata": {
                    "source": "recipe",
                    "uri": str(path.relative_to(RECIPE_IMAGE_DIR.parent.parent.parent)).replace("\\", "/"),
                    "name": recipe.get("name", ""),
                    "cuisine": recipe.get("cuisine", ""),
                    "itemId": -1,
                },
                "_image": ("file", path),
            }
        )
    names = {r["itemId"]: r.get("name", "") for r in restaurants}
    for rv in reviews:
        captions = rv.get("image_captions") or []
        for j, url in enumerate(parse_image_urls(rv.get("images"))):
            chunks.append(
                {
                    "id": f"img-review-{rv['reviewId']}-{j}",
                    "document": captions[j] if j < len(captions) else "",
                    "metadata": {
                        "source": "review",
                        "uri": url,
                        "name": names.get(rv.get("itemId"), ""),
                        "cuisine": "",
                        "itemId": int(rv.get("itemId", -1)),
                    },
                    "_image": ("url", url),
                }
            )
    return chunks


# ---------------------------------------------------------------------------
# Upsert logic
# ---------------------------------------------------------------------------


def content_hash(chunk: dict) -> str:
    payload = json.dumps({"d": chunk["document"], "m": chunk["metadata"]}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection(client, name: str):
    # Embeddings are always supplied explicitly (Jina), so no embedding function.
    return client.get_or_create_collection(
        name, embedding_function=None, configuration={"hnsw": {"space": "cosine"}}
    )


def sync_collection(collection, chunks: list[dict], embed_fn, delete_stale: bool = True) -> dict:
    """Embed only new/changed chunks, upsert them, delete chunks no longer present."""
    existing = collection.get(include=["metadatas"])
    old_hash = {i: (m or {}).get("hash") for i, m in zip(existing["ids"], existing["metadatas"])}

    changed = []
    for chunk in chunks:
        chunk["metadata"]["hash"] = content_hash(chunk)
        if old_hash.get(chunk["id"]) != chunk["metadata"]["hash"]:
            changed.append(chunk)

    if changed:
        vectors = embed_fn(changed)
        collection.upsert(
            ids=[c["id"] for c in changed],
            embeddings=vectors,
            documents=[c["document"] for c in changed],
            metadatas=[c["metadata"] for c in changed],
        )

    stale = []
    if delete_stale:
        wanted = {c["id"] for c in chunks}
        stale = [i for i in old_hash if i not in wanted]
        if stale:
            collection.delete(ids=stale)
    return {"total": len(chunks), "embedded": len(changed), "deleted": len(stale)}


def embed_text_chunks(chunks: list[dict]) -> list[list[float]]:
    return embed_texts([c["document"] for c in chunks], task="retrieval.passage")


def embed_image_chunks(chunks: list[dict]) -> list[list[float]]:
    payloads = []
    for chunk in chunks:
        kind, ref = chunk["_image"]
        payloads.append(image_to_base64(ref) if kind == "file" else ref)
    return embed_images(payloads)


def build(rebuild: bool = False, skip_images: bool = False) -> dict:
    restaurants = load_json(RESTAURANT_DATA_PATH)
    reviews = load_json(REVIEW_DATA_PATH)
    paragraphs = load_map_paragraphs()

    client = get_client()
    names = [RESTAURANTS_COLLECTION, CULINARY_MAP_COLLECTION, REVIEWS_COLLECTION]
    if not skip_images:
        names.append(IMAGES_COLLECTION)
    if rebuild:
        existing = {c.name for c in client.list_collections()}
        for name in names:
            if name in existing:
                client.delete_collection(name)

    report = {}
    plan = [
        (RESTAURANTS_COLLECTION, restaurant_chunks(restaurants), embed_text_chunks),
        (CULINARY_MAP_COLLECTION, map_chunks(restaurants, paragraphs), embed_text_chunks),
        (REVIEWS_COLLECTION, review_chunks(reviews, restaurants), embed_text_chunks),
    ]
    if not skip_images:
        plan.append((IMAGES_COLLECTION, image_chunks(reviews, restaurants), embed_image_chunks))

    for name, chunks, embed_fn in plan:
        print(f"[{name}] {len(chunks)} chunks ...", flush=True)
        report[name] = sync_collection(get_collection(client, name), chunks, embed_fn)
        print(f"[{name}] {report[name]}", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the ChromaDB index with Jina embeddings.")
    parser.add_argument("--rebuild", action="store_true", help="drop collections and re-embed everything")
    parser.add_argument("--skip-images", action="store_true", help="skip the multimodal image collection")
    args = parser.parse_args()
    build(rebuild=args.rebuild, skip_images=args.skip_images)
    return 0


if __name__ == "__main__":
    sys.exit(main())
