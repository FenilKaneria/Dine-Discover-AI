"""Retrieval evaluation: keyword baseline vs BM25 vs dense vs hybrid vs hybrid + rerank.

Query sets
  Set A  hand-written natural-language requests. Relevance is defined by
         rules over the structured fields (cuisine, type, signatures, vibe,
         environment, location, price, rating), so labels are objective and
         reproducible — nobody hand-picks the "right" answers.
  Set B  known-item search: gpt-oss-20b writes one paraphrased request per
         restaurant (without its name); the target restaurant is the only
         relevant item. Cached in eval/queries_b.json so reruns are identical.
  Set C  like Set B, but vague: no names, places, dishes or drinks.
         Sets B and C are split 50/50 into dev/test (seed 42). Fusion weights
         are tuned on dev only (--tune); tables report the test split.
  Set D  hand-written, human-style known-item queries (eval/queries_d.json),
         not generated from the records, so they are less lexically biased.
  Images text-to-image search over the recipe photos with jina-clip-v2:
         query = recipe name, relevant item = that recipe's photo.

Metrics: Recall@1/5/10, Precision@5, MRR, nDCG@10 (binary relevance), p50 latency,
95% bootstrap confidence intervals, and paired bootstrap deltas vs BM25.

Usage:
    python src/evaluate_retrieval.py --tune           # tune RRF weights on dev, then exit
    python src/evaluate_retrieval.py                  # everything (Jina reranker)
    python src/evaluate_retrieval.py --llm-rerank     # also the gpt-oss-120b reranker on a sample
    python src/evaluate_retrieval.py --no-rerank      # skip reranked systems
"""
import argparse
import itertools
import json
import math
import random
import re
import statistics
import sys
import time
from datetime import date

from config import CHAT_MODEL, EMBED_MODEL, EVAL_DIR, IMAGE_EMBED_MODEL, REASONING_MODEL, RERANK_MODEL, groq_client
from embeddings import embed_clip_texts
from retrieval import RRF_WEIGHTS, cached_query_embedding, get_retriever, rrf

K_VALUES = (1, 5, 10)
CI_METRICS = ("mrr", "ndcg@10")
BOOTSTRAP_RESAMPLES = 1000
# system -> (mode, rerank, reranker, RRF weights; None = tuned RRF_WEIGHTS)
SYSTEM_SPECS = {
    "keyword": ("keyword", False, None, None),
    "bm25": ("bm25", False, None, None),
    "dense": ("dense", False, None, None),
    "hybrid (equal weights)": ("hybrid", False, None, (1.0, 1.0, 1.0)),
    "hybrid": ("hybrid", False, None, None),
    "hybrid+rerank": ("hybrid", True, "jina", None),
    "hybrid+llm-rerank": ("hybrid", True, "llm", None),
}
SYSTEMS = list(SYSTEM_SPECS)

# ---------------------------------------------------------------------------
# Set A — (query, text regex over descriptive fields, location regex, predicate)
# ---------------------------------------------------------------------------
SET_A = [
    ("somewhere for a comforting bowl of ramen", r"ramen", None, None),
    ("sushi or omakase night", r"sushi|omakase|nigiri", None, None),
    ("dim sum with dumplings and bao", r"dim sum|dumpling|bao", None, None),
    ("korean barbecue grilled at the table", r"korean bbq|korean barbecue|kbbq|galbi", None, None),
    ("fully vegan plant-based dinner", r"vegan|plant-based", None, None),
    ("wood-fired pizza", r"pizza", None, None),
    ("dry-aged steak dinner", r"steak|ribeye|wagyu", None, None),
    ("fresh oysters on the half shell", r"oyster", None, None),
    ("casual taco joint", r"taco", None, None),
    ("persian kebabs with saffron rice", r"persian|kebab|koobideh|saffron", None, None),
    ("fiery spicy sichuan dishes", r"sichuan|szechuan|mapo|mala", None, None),
    ("vietnamese pho or banh mi", r"vietnam|pho\b|banh mi", None, None),
    ("indian curries and tandoori", r"indian|curry|tandoor|masala|biryani|dosa", None, None),
    ("classic french bistro", r"french|bistro", None, None),
    ("texas style smoked brisket", r"brisket|texas", None, None),
    ("crispy fried chicken comfort food", r"fried chicken", None, None),
    ("farm to table seasonal cooking", r"farm-to-table|farm to table|seasonal", None, None),
    ("romantic candlelit date night", r"romantic|candle", None, None),
    ("dark moody speakeasy", r"speakeasy|moody|dimly", None, None),
    ("calm zen minimalist atmosphere", r"zen|minimalis|tranquil|serene", None, None),
    ("lively buzzing high-energy crowd", r"lively|energetic|buzzing|bustling|raucous|electric", None, None),
    ("family friendly place for kids", r"family", None, None),
    ("late night bite after midnight", r"late-night|late night|midnight|after-hours", None, None),
    ("dining with an ocean view", r"ocean|beach|waterfront|pier|seaside|coastal|harbor", None, None),
    ("garden patio outdoor seating", r"garden|patio|courtyard|outdoor", None, None),
    ("retro diner", r"diner|retro", None, None),
    ("old hollywood glamour", r"hollywood|glamour|glamorous|art deco", None, None),
    ("tasting menu for a big splurge", r"tasting|fine dining|upscale|luxur", None, lambda r: r.get("price_range") == 4),
    ("brunch with pancakes", r"brunch|pancake|waffle", None, None),
    ("mediterranean mezze and hummus", r"hummus|mezze|meze|mediterranean|lebanese|falafel", None, None),
    ("oaxacan mole and mezcal", r"oaxaca|mole|mezcal", None, None),
    ("wine bar with small plates", r"wine bar|small plates|tapas", None, None),
    ("cheap mexican food", r"mexican|taco|burrito|taqueria", None, lambda r: (r.get("price_range") or 9) <= 2),
    ("highly rated italian pasta", r"italian|pasta|trattoria", None, lambda r: (r.get("rating") or 0) >= 4.5),
    ("japanese food in little tokyo", r"japanese|sushi|ramen|izakaya", r"little tokyo", None),
    ("places to eat in pasadena", r".", r"pasadena", None),
    ("seafood in san pedro", r"seafood|fish|oyster|lobster", r"san pedro", None),
    ("wine country dining in napa or healdsburg", r".", r"napa|healdsburg|sonoma|st\. helena|yountville", None),
    ("cozy mountain lodge comfort food", r"mountain|cabin|lodge|alpine", None, None),
    ("gastropub with craft beer", r"gastropub|craft beer|brew|pub", None, None),
]


def descriptive_text(r: dict) -> str:
    parts = [r.get("name", ""), r.get("food_style", ""), r.get("type", ""), r.get("vibe", ""), r.get("environment", "")]
    parts += list(r.get("signatures") or [])
    return " ".join(str(p) for p in parts).lower()


def set_a_labels(records: dict[int, dict]) -> list[dict]:
    queries = []
    for query, text_re, loc_re, pred in SET_A:
        relevant = [
            i
            for i, r in records.items()
            if re.search(text_re, descriptive_text(r))
            and (loc_re is None or re.search(loc_re, str(r.get("location", "")).lower()))
            and (pred is None or pred(r))
        ]
        if relevant:
            queries.append({"query": query, "relevant": relevant})
        else:
            print(f"  [set A] dropped (no relevant records): {query}")
    return queries


# ---------------------------------------------------------------------------
# Set B — LLM-generated known-item queries (cached)
# ---------------------------------------------------------------------------
SET_B_PROMPT = """For each restaurant below, write ONE realistic search request a diner might type
into a restaurant finder when looking for a place like it. Rules:
- Do NOT use the restaurant's name.
- Paraphrase: describe the food, dishes, vibe or setting in your own words; avoid copying exact phrases.
- You may mention the city/neighborhood in about half of the requests.
- 8 to 20 words, casual tone.

Restaurants (JSON):
{restaurants}

Return JSON only: {{"queries": [{{"itemId": <itemId>, "query": "<request>"}}, ...]}}"""


SET_C_PROMPT = """For each restaurant below, write ONE vague search request a diner might type when
they want an experience like this place but don't know it exists. Strict rules:
- Do NOT use the restaurant's name, city, neighborhood or any place name.
- Do NOT name any specific dish, ingredient or drink.
- Describe only the atmosphere, setting, occasion and the broad style of food, in your own words.
- 8 to 20 words, casual tone.

Restaurants (JSON):
{restaurants}

Return JSON only: {{"queries": [{{"itemId": <itemId>, "query": "<request>"}}, ...]}}"""


def generate_set_b(records: dict[int, dict], path, batch_size: int = 10, prompt: str = SET_B_PROMPT) -> list[dict]:
    """Generate (or load cached) known-item queries. Also used for Set C."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    client = groq_client()
    items = list(records.values())
    out: list[dict] = []
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        payload = [
            {k: r.get(k) for k in ("itemId", "name", "location", "type", "food_style", "vibe", "environment", "signatures")}
            for r in batch
        ]
        for attempt in range(4):
            try:
                resp = client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[{"role": "user", "content": prompt.format(restaurants=json.dumps(payload))}],
                    response_format={"type": "json_object"},
                    temperature=0.7,
                    reasoning_effort="low",
                )
                got = {int(q["itemId"]): q["query"] for q in json.loads(resp.choices[0].message.content)["queries"]}
                break
            except Exception as exc:
                print(f"  [set B] batch {start} attempt {attempt + 1} failed: {exc}")
                time.sleep(20)
        else:
            raise RuntimeError("Set B generation failed repeatedly")
        for r in batch:
            if int(r["itemId"]) in got:
                out.append({"itemId": int(r["itemId"]), "query": got[int(r["itemId"])]})
        print(f"  [set B] generated {len(out)}/{len(items)}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def set_b_labels(raw: list[dict], records: dict[int, dict]) -> list[dict]:
    """Relevant = the target, plus any exact duplicate record (same name + location)."""
    queries = []
    for q in raw:
        target = records.get(q["itemId"])
        if not target:
            continue
        relevant = [
            i
            for i, r in records.items()
            if r.get("name") == target.get("name") and r.get("location") == target.get("location")
        ]
        queries.append({"query": q["query"], "relevant": relevant, "target": q["itemId"]})
    return queries


def set_d_labels(records: dict[int, dict]) -> list[dict]:
    """Set D: hand-written known-item queries (eval/queries_d.json)."""
    raw = json.loads((EVAL_DIR / "queries_d.json").read_text(encoding="utf-8"))["known_item"]
    return set_b_labels(raw, records)


def dev_test_split(queries: list[dict], seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Deterministic 50/50 split. Fusion weights are tuned on dev; results are reported on test."""
    shuffled = list(queries)
    random.Random(seed).shuffle(shuffled)
    half = len(shuffled) // 2
    return shuffled[:half], shuffled[half:]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def metrics_for(ranked: list, relevant: set) -> dict:
    m = {}
    for k in K_VALUES:
        m[f"recall@{k}"] = len(set(ranked[:k]) & relevant) / len(relevant)
    for k in (1, 5):
        m[f"hit@{k}"] = 1.0 if set(ranked[:k]) & relevant else 0.0
    m["precision@5"] = len(set(ranked[:5]) & relevant) / 5
    rr = 0.0
    for rank, item in enumerate(ranked, start=1):
        if item in relevant:
            rr = 1.0 / rank
            break
    m["mrr"] = rr
    dcg = sum(1.0 / math.log2(rank + 1) for rank, item in enumerate(ranked[:10], start=1) if item in relevant)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), 10) + 1))
    m["ndcg@10"] = dcg / idcg if idcg else 0.0
    return m


def bootstrap_ci(values: list[float], n_resamples: int = BOOTSTRAP_RESAMPLES, seed: int = 0) -> list[float]:
    """95% percentile bootstrap confidence interval of the mean."""
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_resamples))
    return [round(means[int(0.025 * n_resamples)], 4), round(means[int(0.975 * n_resamples) - 1], 4)]


def aggregate(per_query: list[dict], latencies: list[float]) -> dict:
    keys = per_query[0].keys()
    out = {k: round(statistics.mean(q[k] for q in per_query), 4) for k in keys}
    for k in CI_METRICS:
        out[f"{k}_ci95"] = bootstrap_ci([q[k] for q in per_query])
    out["n_queries"] = len(per_query)
    out["latency_p50_ms"] = round(statistics.median(latencies) * 1000, 1)
    return out


def paired_delta_ci(a: list[dict], b: list[dict], metric: str = "ndcg@10") -> dict:
    """Mean difference a - b on the same queries, with a paired bootstrap 95% CI.
    If the interval excludes 0, the difference is unlikely to be noise."""
    diffs = [x[metric] - y[metric] for x, y in zip(a, b)]
    return {"metric": metric, "mean_delta": round(statistics.mean(diffs), 4), "ci95": bootstrap_ci(diffs)}


def run_system(retriever, queries: list[dict], system: str) -> tuple[dict, list[dict]]:
    mode, rerank, reranker, weights = SYSTEM_SPECS[system]
    per_query, latencies, failures = [], [], 0
    for q in queries:
        cached_query_embedding.cache_clear()  # latency includes the query-embedding call
        for attempt in range(3):
            try:
                t0 = time.perf_counter()
                ranked = retriever.ranked_ids(
                    q["query"], mode=mode, rerank=rerank, n=10, reranker=reranker, weights=weights
                )
                latencies.append(time.perf_counter() - t0)
                per_query.append(metrics_for(ranked, set(q["relevant"])))
                break
            except Exception as exc:
                print(f"    [{system}] retry {attempt + 1}: {type(exc).__name__}: {str(exc)[:120]}")
                time.sleep(30)
        else:
            failures += 1
            per_query.append(None)
    ok = [m for m in per_query if m is not None]
    result = aggregate(ok, latencies) if ok else {"n_queries": 0}
    result["failed_queries"] = failures
    return result, per_query


# ---------------------------------------------------------------------------
# Fusion weight tuning (dev split only)
# ---------------------------------------------------------------------------


def tune_fusion(retriever, dev_queries: list[dict]) -> dict:
    """Grid-search RRF weights for [dense, dense_map, bm25] by mean nDCG@10 on dev.
    The three ranked lists are computed once per query, so the grid costs no API calls."""
    lists = []
    for q in dev_queries:
        for attempt in range(3):
            try:
                lists.append((retriever.fusion_lists(q["query"]), set(q["relevant"])))
                break
            except Exception as exc:
                print(f"    [tune] retry {attempt + 1}: {exc}")
                time.sleep(30)
    grid = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)
    scores = []
    for weights in itertools.product(grid, repeat=3):
        if not any(weights):
            continue
        per_query = [metrics_for([i for i, _ in rrf(ls, weights=weights)], rel) for ls, rel in lists]
        scores.append(
            {
                "weights": list(weights),
                "ndcg@10": round(statistics.mean(m["ndcg@10"] for m in per_query), 4),
                "mrr": round(statistics.mean(m["mrr"] for m in per_query), 4),
            }
        )
    scores.sort(key=lambda s: (s["ndcg@10"], s["mrr"]), reverse=True)
    equal = next(s for s in scores if s["weights"] == [1.0, 1.0, 1.0])
    # Sets B/C are written from the record text, so BM25 alone tends to win on
    # them; best_hybrid keeps every retriever for queries that share no words.
    best_hybrid = next(s for s in scores if all(s["weights"]))
    return {
        "n_dev_queries": len(lists),
        "best": scores[0],
        "best_hybrid": best_hybrid,
        "equal_weights": equal,
        "top10": scores[:10],
    }


# ---------------------------------------------------------------------------
# Image retrieval
# ---------------------------------------------------------------------------


def evaluate_images(retriever) -> dict:
    """Text-to-image: recipe name -> its own photo, among all indexed images."""
    col = retriever.images_col
    got = col.get(where={"source": "recipe"}, include=["metadatas"])
    targets = [(i, m["name"]) for i, m in zip(got["ids"], got["metadatas"])]
    vectors = embed_clip_texts([name for _, name in targets])
    per_query = []
    for (image_id, _), vec in zip(targets, vectors):
        res = col.query(query_embeddings=[vec], n_results=10)
        per_query.append(metrics_for(res["ids"][0], {image_id}))
    out = {k: round(statistics.mean(q[k] for q in per_query), 4) for k in ("recall@1", "recall@5", "recall@10", "mrr")}
    out["n_queries"] = len(per_query)
    out["n_images_indexed"] = col.count()
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def fmt(m: dict, col: str) -> str:
    value = m.get(col, "-")
    ci = m.get(f"{col}_ci95")
    return f"{value} [{ci[0]}–{ci[1]}]" if ci else str(value)


def markdown_table(results: dict) -> str:
    lines = []
    for set_name, systems in sorted(results["text"].items()):
        if set_name.startswith("Set A"):
            cols = ["recall@10", "precision@5", "mrr", "ndcg@10", "latency_p50_ms"]
        else:  # known-item sets: did the target show up, and how high?
            cols = ["hit@1", "hit@5", "mrr", "ndcg@10", "latency_p50_ms"]
        n = max(s.get("n_queries", 0) for s in systems.values())
        lines.append(f"\n**{set_name}** ({n} queries; [95% bootstrap CI])\n")
        lines.append("| System | " + " | ".join(c.replace("_", " ") for c in cols) + " |")
        lines.append("|---" * (len(cols) + 1) + "|")
        for system, m in systems.items():
            label = system + (f" (n={m['n_queries']})" if m.get("n_queries") != n else "")
            if m.get("failed_queries"):
                label += f" [{m['failed_queries']} failed]"
            lines.append(f"| {label} | " + " | ".join(fmt(m, c) for c in cols) + " |")
        deltas = results.get("paired_deltas", {}).get(set_name)
        if deltas:
            lines.append("")
            for name, d in deltas.items():
                lines.append(f"- {name}: Δ{d['metric']} = {d['mean_delta']:+} (95% CI {d['ci95'][0]} to {d['ci95'][1]})")
    if "images" in results:
        im = results["images"]
        lines.append(f"\n**Text-to-image** ({im['n_queries']} recipe-name queries over {im['n_images_indexed']} images)\n")
        lines.append("| Recall@1 | Recall@5 | Recall@10 | MRR |")
        lines.append("|---|---|---|---|")
        lines.append(f"| {im['recall@1']} | {im['recall@5']} | {im['recall@10']} | {im['mrr']} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tune", action="store_true", help="grid-search RRF weights on the dev split and exit")
    parser.add_argument("--no-rerank", action="store_true", help="skip the reranked systems")
    parser.add_argument("--llm-rerank", action="store_true",
                        help="also run the gpt-oss-120b listwise reranker (slow; Groq free tier: 8k tokens/min)")
    parser.add_argument("--llm-rerank-sample", type=int, default=40,
                        help="queries per set for the LLM reranker")
    parser.add_argument("--sets", default="A,B,C,D",
                        help="query sets to (re)run; other sets are kept from eval/results.json")
    parser.add_argument("--skip-images", action="store_true")
    args = parser.parse_args()
    wanted = {s.strip().upper() for s in args.sets.split(",")}

    retriever = get_retriever()
    records = retriever.records
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    print("Building query sets ...")
    set_b_dev, set_b_test = dev_test_split(set_b_labels(generate_set_b(records, EVAL_DIR / "queries_b.json"), records))
    set_c_dev, set_c_test = dev_test_split(
        set_b_labels(generate_set_b(records, EVAL_DIR / "queries_c.json", prompt=SET_C_PROMPT), records)
    )

    if args.tune:
        report = tune_fusion(retriever, set_b_dev + set_c_dev)
        (EVAL_DIR / "fusion_tuning.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({k: report[k] for k in ("n_dev_queries", "best", "best_hybrid", "equal_weights")}, indent=2))
        print("Set RRF_WEIGHTS in src/retrieval.py (best_hybrid), then rerun without --tune.")
        return 0

    query_sets = []
    if "A" in wanted:
        query_sets.append(("Set A — rule-labelled requests", set_a_labels(records)))
    if "B" in wanted:
        query_sets.append(("Set B — known-item paraphrases (test split)", set_b_test))
    if "C" in wanted:
        query_sets.append(("Set C — vague known-item, no names/places/dishes (test split)", set_c_test))
    if "D" in wanted:
        query_sets.append(("Set D — hand-written human-style known-item", set_d_labels(records)))

    previous_path = EVAL_DIR / "results.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8")) if previous_path.exists() else {}

    results = {
        "date": date.today().isoformat(),
        "config": {
            "embedding_model": EMBED_MODEL,
            "image_embedding_model": IMAGE_EMBED_MODEL,
            "reranker": RERANK_MODEL,
            "llm_reranker": REASONING_MODEL,
            "query_generator": CHAT_MODEL,
            "vector_db": "ChromaDB (cosine HNSW)",
            "fusion": f"weighted RRF k=60, weights {list(RRF_WEIGHTS)} over dense(restaurants), dense(culinary map), BM25",
            "corpus_restaurants": len(records),
            "rerank_candidates": 20,
            "split": "Sets B and C: 50/50 dev/test (seed 42); fusion tuned on dev, reported on test",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        },
        # Sets not re-run this time are carried over from the previous results.
        "text": {
            name: systems
            for name, systems in previous.get("text", {}).items()
            if name.split(" ")[1] not in wanted and " — " in name
        },
        "paired_deltas": {
            name: d for name, d in previous.get("paired_deltas", {}).items() if name.split(" ")[1] not in wanted
        },
    }

    for set_name, queries in query_sets:
        results["text"][set_name] = {}
        per_system: dict[str, list] = {}
        for system in SYSTEMS:
            if "rerank" in system and args.no_rerank:
                continue
            if system == "hybrid+llm-rerank":
                if not args.llm_rerank:
                    continue
                qs = random.Random(42).sample(queries, min(args.llm_rerank_sample, len(queries)))
            else:
                qs = queries
            print(f"[{set_name[:5]}] {system}: {len(qs)} queries ...", flush=True)
            summary, per_query = run_system(retriever, qs, system)
            results["text"][set_name][system] = summary
            per_system[system] = per_query
            print(f"    ndcg@10={summary.get('ndcg@10')} mrr={summary.get('mrr')} p50={summary.get('latency_p50_ms')} ms", flush=True)

        # Paired comparisons on identical queries (skip queries that failed in either system).
        deltas = {}
        for system in ("hybrid", "hybrid+rerank"):
            if system in per_system and "bm25" in per_system:
                pairs = [(a, b) for a, b in zip(per_system[system], per_system["bm25"]) if a and b]
                deltas[f"{system} − bm25"] = paired_delta_ci([a for a, _ in pairs], [b for _, b in pairs])
        if "hybrid" in per_system and "hybrid (equal weights)" in per_system:
            pairs = [(a, b) for a, b in zip(per_system["hybrid"], per_system["hybrid (equal weights)"]) if a and b]
            deltas["hybrid (tuned) − hybrid (equal weights)"] = paired_delta_ci([a for a, _ in pairs], [b for _, b in pairs])
        results["paired_deltas"][set_name] = deltas

    if not args.skip_images:
        print("Image retrieval ...", flush=True)
        results["images"] = evaluate_images(retriever)
    elif "images" in previous:
        results["images"] = previous["images"]

    (EVAL_DIR / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    table = markdown_table(results)
    header = (
        f"# Retrieval evaluation ({results['date']})\n\n"
        f"Embeddings `{EMBED_MODEL}` (text) / `{IMAGE_EMBED_MODEL}` (images), ChromaDB, "
        f"reranker `{RERANK_MODEL}`, RRF weights {list(RRF_WEIGHTS)}, corpus {len(records)} restaurants.\n"
    )
    (EVAL_DIR / "results.md").write_text(header + table + "\n", encoding="utf-8")
    print(table)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # tables use non-ASCII (Δ, –)
    sys.exit(main())
