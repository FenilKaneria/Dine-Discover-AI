"""Hybrid retrieval over the ChromaDB index.

Pipeline for restaurant recommendations:
  1. Dense   — Jina query embedding vs. `restaurants` collection (Chroma, cosine)
  2. Dense   — same query vs. `culinary_map` prose paragraphs, mapped to itemId
  3. Sparse  — BM25 over each restaurant's document + its culinary-map paragraph
  4. Fusion  — weighted Reciprocal Rank Fusion (RRF, k=60) of the three lists;
               weights tuned on the dev split (`evaluate_retrieval.py --tune`)
  5. Filters — location / max price / min rating (price & rating pushed down
               to Chroma `where`; location applied as a substring match)
  6. Rerank  — optional rerank of the top-N: Jina cross-encoder (default) or
               an LLM listwise rerank with gpt-oss-120b (RERANKER=llm)

`keyword_search` reproduces the original substring matcher and is kept only as
the evaluation baseline.
"""
import json
import re
from functools import lru_cache

from rank_bm25 import BM25Okapi

from build_index import get_client, get_collection, load_json
from config import (
    CULINARY_MAP_COLLECTION,
    IMAGES_COLLECTION,
    REASONING_MODEL,
    RERANKER,
    RESTAURANT_DATA_PATH,
    RESTAURANTS_COLLECTION,
    REVIEWS_COLLECTION,
    groq_client,
)
from embeddings import embed_clip_texts, embed_images, embed_query, rerank as jina_rerank

RRF_K = 60
# Weights for [dense(restaurants), dense(culinary map), BM25], tuned on the
# Set B+C dev split (eval/fusion_tuning.json, best with every retriever kept).
RRF_WEIGHTS = (0.25, 0.25, 2.0)
STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "for", "to", "with", "at", "is", "are",
    "i", "me", "my", "we", "want", "looking", "find", "some", "place", "spot", "restaurant",
    "restaurants", "somewhere", "good", "that", "has", "have", "like", "near", "where", "can",
}


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9$]+", text.lower()) if t not in STOPWORDS]


@lru_cache(maxsize=2048)
def cached_query_embedding(query: str) -> tuple[float, ...]:
    """Queries repeat across tools and eval systems; embed each one once."""
    return tuple(embed_query(query))


def rrf(
    ranked_lists: list[list[int]], k: int = RRF_K, weights: tuple[float, ...] | None = None
) -> list[tuple[int, float]]:
    """Weighted Reciprocal Rank Fusion: score(d) = sum over lists of w / (k + rank)."""
    weights = weights or (1.0,) * len(ranked_lists)
    scores: dict[int, float] = {}
    for ranked, w in zip(ranked_lists, weights):
        if not w:
            continue
        for rank, item in enumerate(ranked, start=1):
            scores[item] = scores.get(item, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


class HybridRetriever:
    def __init__(self):
        client = get_client()
        self.restaurants_col = get_collection(client, RESTAURANTS_COLLECTION)
        self.map_col = get_collection(client, CULINARY_MAP_COLLECTION)
        self.reviews_col = get_collection(client, REVIEWS_COLLECTION)
        self.images_col = get_collection(client, IMAGES_COLLECTION)
        if self.restaurants_col.count() == 0:
            raise RuntimeError("Vector index is empty. Run `python src/build_index.py` first.")

        self.records = {int(r["itemId"]): r for r in load_json(RESTAURANT_DATA_PATH)}

        # Documents come from Chroma so BM25 and dense search see identical text.
        rest = self.restaurants_col.get(include=["documents", "metadatas"])
        self.documents = {int(m["itemId"]): d for d, m in zip(rest["documents"], rest["metadatas"])}
        maps = self.map_col.get(include=["documents", "metadatas"])
        self.map_text: dict[int, str] = {}
        for doc, meta in zip(maps["documents"], maps["metadatas"]):
            if int(meta["itemId"]) >= 0:
                self.map_text[int(meta["itemId"])] = doc

        self.bm25_ids = list(self.documents)
        self.bm25 = BM25Okapi(
            [tokenize(self.documents[i] + " " + self.map_text.get(i, "")) for i in self.bm25_ids]
        )

    # -- filters ------------------------------------------------------------

    @staticmethod
    def _where(max_price: int | None, min_rating: float | None) -> dict | None:
        clauses = []
        if max_price:
            clauses.append({"price_range": {"$lte": int(max_price)}})
        if min_rating:
            clauses.append({"rating": {"$gte": float(min_rating)}})
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def _passes(self, item_id: int, location, max_price, min_rating) -> bool:
        r = self.records.get(item_id)
        if r is None:
            return False
        if location and location.lower() not in str(r.get("location", "")).lower():
            return False
        if max_price and (r.get("price_range") or 0) > int(max_price):
            return False
        if min_rating and (r.get("rating") or 0) < float(min_rating):
            return False
        return True

    # -- individual retrievers ---------------------------------------------

    def dense(self, query: str, n: int = 50, where: dict | None = None) -> list[int]:
        res = self.restaurants_col.query(
            query_embeddings=[list(cached_query_embedding(query))],
            n_results=min(n, self.restaurants_col.count()),
            where=where,
        )
        return [int(m["itemId"]) for m in res["metadatas"][0]]

    def dense_map(self, query: str, n: int = 50) -> list[int]:
        res = self.map_col.query(
            query_embeddings=[list(cached_query_embedding(query))],
            n_results=min(n, self.map_col.count()),
        )
        return [int(m["itemId"]) for m in res["metadatas"][0] if int(m["itemId"]) >= 0]

    def sparse(self, query: str, n: int = 50) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self.bm25_ids, scores), key=lambda kv: kv[1], reverse=True)
        return [i for i, s in ranked[:n] if s > 0]

    def keyword_search(self, query: str, n: int = 50) -> list[int]:
        """Original substring matcher (baseline only): phrase match, then any-word match."""
        q = query.lower().strip()

        def hay(r):
            return " ".join(
                str(r.get(k, "")) for k in ("vibe", "environment", "type", "food_style", "location")
            ).lower()

        hits = [r for r in self.records.values() if q in hay(r)]
        if not hits:
            words = [w for w in re.split(r"[\s,]+", q) if len(w) >= 3]
            hits = [r for r in self.records.values() if any(w in hay(r) for w in words)]
        hits.sort(key=lambda r: r.get("rating") or 0, reverse=True)
        return [int(r["itemId"]) for r in hits[:n]]

    # -- full pipeline ------------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 5,
        location: str | None = None,
        max_price: int | None = None,
        min_rating: float | None = None,
        rerank: bool = False,
        rerank_candidates: int = 20,
        mode: str = "hybrid",
        strict_rerank: bool = False,
        reranker: str | None = None,
        weights: tuple[float, ...] | None = None,
    ) -> list[dict]:
        """Return the top-k restaurants as dicts with itemId, score and evidence.

        mode: "hybrid" (default), "dense", "bm25" or "keyword".
        reranker: "jina" or "llm" (default from the RERANKER env var).
        """
        where = self._where(max_price, min_rating)
        if mode == "keyword":
            fused = [(i, 0.0) for i in self.keyword_search(query, n=len(self.records))]
        elif mode == "bm25":
            fused = [(i, 0.0) for i in self.sparse(query, n=len(self.records))]
        elif mode == "dense":
            fused = [(i, 0.0) for i in self.dense(query, n=len(self.records), where=where)]
        else:
            fused = rrf(self.fusion_lists(query, where), weights=weights or RRF_WEIGHTS)

        candidates = [
            (i, s) for i, s in fused if self._passes(i, location, max_price, min_rating)
        ]
        if rerank and candidates:
            head = [i for i, _ in candidates[:rerank_candidates]]
            tail = candidates[rerank_candidates:]
            if (reranker or RERANKER) == "llm":
                order = llm_rerank(query, [(i, self.documents.get(i, "")) for i in head], strict=strict_rerank)
                candidates = [(i, 1.0 / (1 + pos)) for pos, i in enumerate(order)] + tail
            else:
                candidates = self.cross_encoder_rerank(query, head, strict=strict_rerank) + tail

        results = []
        for item_id, score in candidates[:k]:
            results.append(
                {
                    "itemId": item_id,
                    "score": round(float(score), 5),
                    "record": self.records[item_id],
                    "evidence": self.map_text.get(item_id, self.documents.get(item_id, "")),
                }
            )
        return results

    def fusion_lists(self, query: str, where: dict | None = None, n: int = 100) -> list[list[int]]:
        """The three ranked lists RRF fuses, in RRF_WEIGHTS order."""
        return [self.dense(query, n=n, where=where), self.dense_map(query, n=n), self.sparse(query, n=n)]

    def cross_encoder_rerank(self, query: str, head: list[int], strict: bool = False) -> list[tuple[int, float]]:
        """Rerank candidates with the Jina cross-encoder over record + prose text.
        Falls back to the fused order on failure (strict=True raises instead)."""
        docs = [(self.documents.get(i, "") + "\n" + self.map_text.get(i, "")).strip() for i in head]
        try:
            scored = jina_rerank(query, docs)
        except Exception:
            if strict:
                raise
            return [(i, 0.0) for i in head]
        ordered = [(head[idx], score) for idx, score in scored]
        seen = {i for i, _ in ordered}
        return ordered + [(i, 0.0) for i in head if i not in seen]

    def ranked_ids(
        self,
        query: str,
        mode: str,
        rerank: bool = False,
        n: int = 10,
        reranker: str | None = None,
        weights: tuple[float, ...] | None = None,
    ) -> list[int]:
        """Ranked itemIds only — used by the evaluation harness."""
        hits = self.search(
            query, k=n, mode=mode, rerank=rerank, strict_rerank=True, reranker=reranker, weights=weights
        )
        return [r["itemId"] for r in hits]

    def resolve_name(self, name: str, k: int = 3) -> list[int]:
        """Semantic fallback when a restaurant name has no string match (typos etc.)."""
        return self.dense(name, n=k)

    # -- knowledge base + images --------------------------------------------

    def search_passages(self, query: str, k: int = 5) -> list[dict]:
        """Dense search over culinary-map prose and reviews, merged by distance."""
        vec = [list(cached_query_embedding(query))]
        hits = []
        for source, col in (("culinary_map", self.map_col), ("review", self.reviews_col)):
            res = col.query(query_embeddings=vec, n_results=min(k, col.count()))
            for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
                hits.append(
                    {
                        "source": source,
                        "restaurant": meta.get("name", ""),
                        "itemId": meta.get("itemId"),
                        "similarity": round(1 - float(dist), 4),
                        "text": doc,
                    }
                )
        hits.sort(key=lambda h: h["similarity"], reverse=True)
        return hits[:k]

    def search_images(self, query: str | None = None, image_b64: str | None = None, k: int = 4) -> list[dict]:
        """Text-to-image or image-to-image search in the jina-clip-v2 space."""
        if image_b64:
            vec = embed_images([image_b64])[0]
        else:
            vec = embed_clip_texts([query or ""])[0]
        res = self.images_col.query(query_embeddings=[vec], n_results=min(k, self.images_col.count()))
        return [
            {
                "uri": meta["uri"],
                "source": meta["source"],
                "name": meta.get("name", ""),
                "description": doc,
                "similarity": round(1 - float(dist), 4),
            }
            for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]


RERANK_PROMPT = """You are a search relevance judge for a California restaurant finder.
Rank the candidate restaurants by how well each satisfies the user's request.
Consider cuisine, dishes, vibe, setting, location, price and rating.

Request: {query}

Candidates:
{candidates}

Return JSON only: {{"ranking": [<candidate ids, best first, include every id exactly once>]}}"""


def llm_rerank(query: str, candidates: list[tuple[int, str]], strict: bool = False) -> list[int]:
    """Listwise LLM rerank. Falls back to the input order on any failure
    (strict=True raises instead, so evaluation never counts a silent fallback)."""
    original = [i for i, _ in candidates]
    listing = "\n".join(f"[{i}] {doc}" for i, doc in candidates)
    try:
        resp = groq_client().chat.completions.create(
            model=REASONING_MODEL,
            messages=[{"role": "user", "content": RERANK_PROMPT.format(query=query, candidates=listing)}],
            response_format={"type": "json_object"},
            temperature=0,
            reasoning_effort="low",
        )
        ranking = json.loads(resp.choices[0].message.content)["ranking"]
        ranking = [int(i) for i in ranking if int(i) in original]
    except Exception:
        if strict:
            raise
        return original
    seen = set()
    ordered = [i for i in ranking if not (i in seen or seen.add(i))]
    return ordered + [i for i in original if i not in seen]


_RETRIEVER: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = HybridRetriever()
    return _RETRIEVER
