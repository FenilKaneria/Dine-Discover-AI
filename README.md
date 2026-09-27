# Dine-Discover-AI

Dine-Discover-AI is a conversational **RAG agent** that acts as an expert guide to California's restaurant scene. Ask in plain language — *"cheap tacos near the beach"*, *"a calm zen place for sushi"*, *"what did reviewers say about The Gilded Artichoke?"*, *"show me a margherita pizza"* — and the agent retrieves the relevant restaurants, reviews and photos from a vector database, then writes an answer grounded only in what it retrieved.

It is built on the **Model Context Protocol (MCP)**: retrieval is exposed as MCP tools, and a ReAct agent decides which tools to call.

## Features

- **Hybrid retrieval (RAG):** dense vector search (Jina embeddings in ChromaDB) plus BM25 keyword search, fused with weighted Reciprocal Rank Fusion (weights tuned on a dev split).
- **Cross-encoder reranking:** `jina-reranker-v2-base-multilingual` reorders the top 20 candidates (about 0.8 s). The older `gpt-oss-120b` listwise reranker is still available with `RERANKER=llm`.
- **Grounded generation with citations:** a fast model (`gpt-oss-20b`) routes tool calls, and a stronger model (`gpt-oss-120b`) writes the final answer from the retrieved context only, citing sources as `[n]`.
- **RAG trace panel:** under each answer, the UI shows which MCP tools were called, their arguments and latency, the retrieval method, and the numbered sources the answer cites.
- **Metadata filters:** location, maximum price ($–$$$$) and minimum rating. Price and rating filters are applied inside Chroma's `where` clause.
- **Multimodal search:** `jina-clip-v2` puts text and images in one vector space. You can search for photos by description, or upload a food photo to find similar dishes.
- **Semantic fallbacks:** misspelled or descriptive restaurant names still resolve, via embedding search.
- **Measured quality, end to end:** retrieval (Recall@k, MRR, nDCG@10 with bootstrap CIs), agent routing (tool and argument accuracy), and answers (faithfulness, hallucinated names, citation validity, declining out-of-corpus requests). See [Evaluation results](#evaluation-results).

## Architecture

```text
                      ┌──────────────── offline: python src/build_index.py ────────────────┐
 data/processed/*.json │  chunk → embed (Jina v3 / jina-clip-v2) → ChromaDB (data/chroma/)  │
 data/raw/*.txt, *.png └────────────────────────────────────────────────────────────────────┘

Browser ──► Gradio UI (src/app.py) ── MCP host + ReAct loop
                 │
                 ├──► gpt-oss-20b  (Groq)  — decides which retrieval tools to call
                 ├──► gpt-oss-120b (Groq)  — writes the grounded final answer
                 │
                 └──► MCP server (src/server.py, stdio subprocess)
                          ├─ recommend_by_vibe      → HybridRetriever (dense + BM25 → weighted RRF → Jina rerank)
                          ├─ search_knowledge_base  → dense search over culinary-map prose + reviews
                          ├─ search_images          → jina-clip-v2 text→image search
                          ├─ get_restaurant_info    → name match, semantic fallback
                          ├─ get_review             → itemId join, semantic fallback, photo URLs
                          └─ resource: culinary-map://california
```

### Retrieval pipeline (`src/retrieval.py`)

1. **Dense (restaurants):** the query is embedded with `jina-embeddings-v3` (`task=retrieval.query`) and matched by cosine similarity against one document per restaurant, which were embedded with `task=retrieval.passage`.
2. **Dense (culinary map):** the same query is matched against the prose paragraph written about each restaurant, and hits are mapped back to the restaurant by `itemId`.
3. **Sparse:** BM25 over each restaurant's document plus its prose paragraph.
4. **Fusion:** weighted Reciprocal Rank Fusion, `score = Σ wᵢ/(60 + rank)`, over the three ranked lists. Weights `(0.25, 0.25, 2.0)` for (dense restaurants, dense culinary map, BM25) were chosen by grid search on the Set B+C dev split (`python src/evaluate_retrieval.py --tune`, saved in `eval/fusion_tuning.json`). BM25 alone scored highest on that split, but it was not chosen, because Sets B and C are written from the record text and so favour exact word overlap; the chosen weights are the best setting that keeps all three retrievers.
5. **Filters:** location, `max_price`, `min_rating`.
6. **Rerank:** the Jina cross-encoder scores the top 20 against each restaurant's record plus its prose paragraph. With `RERANKER=llm`, `gpt-oss-120b` reranks the list instead. If the rerank call fails, the fused order is kept.

### Vector index (`src/build_index.py`)

Chunking is entity-level: one chunk per restaurant record, per culinary-map paragraph and per review. Every source item is a single short paragraph, so it is never split further. Other strategies (merged record + prose chunks, per-field vectors) have not been evaluated yet.

| Collection     | Chunks | Content                                              | Embedding model      |
|----------------|--------|------------------------------------------------------|----------------------|
| `restaurants`  | 210    | one document per structured record + filter metadata | `jina-embeddings-v3` |
| `culinary_map` | 210    | one prose paragraph per restaurant (by itemId)       | `jina-embeddings-v3` |
| `reviews`      | 10     | review text + photo captions                         | `jina-embeddings-v3` |
| `images`       | 118    | 109 recipe photos + 9 review photos                  | `jina-clip-v2`       |

Indexing is incremental: every chunk stores a content hash, so a rerun only re-embeds chunks that changed. Adding, editing or deleting a restaurant with `restaurant_data_management.py` updates the index straight away.

## Project Structure

```text
Dine-Discover-AI/
├── .env                          # API keys (not committed)
├── requirements.txt
├── src/
│   ├── app.py                    # Gradio UI + ReAct agent (MCP host)
│   ├── server.py                 # MCP server: RAG tools + resource
│   ├── client.py                 # MCP smoke test for all tools (no chat LLM)
│   ├── config.py                 # env, model names, paths
│   ├── embeddings.py             # Jina embeddings client (text + multimodal)
│   ├── build_index.py            # ingest → chunk → embed → ChromaDB
│   ├── retrieval.py              # hybrid retriever, RRF, LLM reranker
│   ├── evaluate_retrieval.py     # retrieval metrics
│   ├── evaluate_answers.py       # answer quality: LLM judge + deterministic checks
│   ├── evaluate_agent.py         # tool-routing and argument-extraction accuracy
│   └── restaurant_data_management.py  # CRUD CLI with LLM extraction + index sync
├── eval/                         # evaluation queries and results
└── data/
    ├── raw/                      # culinary map text, recipe photos
    ├── processed/                # structured restaurants, reviews, recipes
    └── chroma/                   # persisted vector index (generated)
```

## Setup & Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/FenilKaneria/Dine-Discover-AI.git
   cd Dine-Discover-AI
   ```

2. **Create a virtual environment** (recommended):

   ```bash
   python -m venv .venv
   .venv\Scripts\activate       # Windows
   source .venv/bin/activate    # macOS / Linux
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Add API keys** to a `.env` file in the project root:

   ```env
   GROQ_API_KEY=gsk_...
   JINA_API_KEY=jina_...
   ```

   `OPENAI_API_KEY` is also accepted in place of `GROQ_API_KEY`. The Groq endpoint is fixed in `src/config.py`, so no base URL is needed.

   > Do not wrap values in quotes, and never leave a quote unterminated. `python-dotenv` silently skips a malformed line *and the line after it*.

5. **Build the vector index** (one-off; takes about a minute and a half on the Jina free tier):

   ```bash
   python src/build_index.py              # incremental
   python src/build_index.py --rebuild    # re-embed everything
   ```

## Usage

```bash
python src/app.py
```

Open <http://127.0.0.1:7860>. Photos found by the agent show up in the chat. Open **"How this answer was built (RAG trace)"** to see the MCP tool calls and sources behind the last answer, and **"Find similar dishes from a photo"** to search by image.

Optional environment overrides:

| Variable            | Default               | Effect                                              |
|---------------------|-----------------------|-----------------------------------------------------|
| `CHAT_MODEL`        | `openai/gpt-oss-20b`  | Tool-routing model and Set B/C query generator.     |
| `REASONING_MODEL`   | `openai/gpt-oss-120b` | Answer generation, LLM reranker, LLM judge.         |
| `EMBED_MODEL`       | `jina-embeddings-v3`  | Text embedding model.                               |
| `IMAGE_EMBED_MODEL` | `jina-clip-v2`        | Multimodal embedding model.                         |
| `RERANKER`          | `jina`                | `jina` (cross-encoder) or `llm` (`gpt-oss-120b`).   |
| `RERANK_MODEL`      | `jina-reranker-v2-base-multilingual` | Jina reranker model.                 |
| `RERANK`            | `true`                | Set to `false` to skip reranking.                   |
| `SHARE`             | `false`               | Set to `true` for a public Gradio link.             |
| `SERVER_NAME`       | `127.0.0.1`           | Set to `0.0.0.0` to expose the UI on your network.  |

**MCP smoke test** (spawns the server over stdio, checks the 5 tools and the resource, then calls each tool once):

```bash
python src/client.py
```

**Restaurant database CLI:**

```bash
python src/restaurant_data_management.py          # interactive CRUD (add uses the LLM; syncs the index)
python src/restaurant_data_management.py --test   # offline unit tests
```

## Evaluation

A RAG agent can fail at three stages, so each one is measured separately:

| Stage | Question | Script |
|---|---|---|
| Retrieval | Are the right restaurants in the top results? | `python src/evaluate_retrieval.py` |
| Agent routing | Does the router pick the right MCP tool and extract the right filters? | `python src/evaluate_agent.py` |
| Generation | Is the answer grounded, cited, and does it decline when nothing matches? | `python src/evaluate_answers.py` |

```bash
python src/evaluate_retrieval.py --tune   # grid-search fusion weights on the dev split
python src/evaluate_retrieval.py          # retrieval metrics → eval/results.{json,md}
python src/evaluate_agent.py              # routing accuracy  → eval/agent_routing.{json,md}
python src/evaluate_answers.py            # answer quality    → eval/answer_quality.{json,md}
```

**Query sets**

- **Set A: rule-labelled requests (40 queries).** Hand-written requests such as *"cheap mexican food"*. A restaurant counts as relevant when its structured fields match a rule (cuisine/dish/vibe regex, location, price, rating). Many restaurants are relevant per query, so Precision@5 and nDCG@10 are the useful columns.
- **Set B: known-item paraphrases.** `gpt-oss-20b` writes one request per restaurant without using its name; the target is the only relevant item.
- **Set C: vague known-item.** Like Set B, but with no names, places, dishes or drinks.
- **Set D: hand-written, human-style known-item (25 queries) plus 15 out-of-corpus requests** (`eval/queries_d.json`). These are not generated from the record text, so they check that the numbers hold up beyond LLM-written queries. The out-of-corpus requests (*"Chicago deep-dish pizza"*, *"kosher deli"*) have no answer in the data. The only correct response is to say so.
- **Dev/test split.** Sets B and C are split 50/50 (seed 42). Fusion weights are tuned on dev only, and every table reports the test half.
- **Confidence intervals.** Bracketed values are 95% bootstrap intervals (1000 resamples). Differences between systems are paired bootstrap deltas on the same queries; if the interval contains 0, the difference may be noise.
- **Agent routing (30 messages, `eval/queries_agent.json`).** Name lookups, reviews, vibe searches with and without filters, open questions, image requests and small talk. Checks the first tool call and the extracted arguments, for example *"under $$"* → `max_price=2`, and no invented filters.
- **Answer quality (20 in-corpus + 10 out-of-corpus).** Each query goes through the production tool, context builder and answer prompt. `gpt-oss-120b` judges faithfulness, relevance and whether the answer declined. Two checks need no LLM: restaurant names in the answer that are not in the retrieved context, and whether each `[n]` citation points to a real source.

## Evaluation results

Measured on 2026-09-27 over the 210-restaurant corpus. Raw numbers are in `eval/`. Latencies are end-to-end from a laptop and include the Jina API call that embeds the query (about 0.75 s).

### Retrieval: nDCG@10 on each set [95% CI]

| System | Set A (40) | Set B test (103) | Set C test (104) | Set D (25) | p50 latency |
|---|---|---|---|---|---|
| keyword (original) | 0.336 | 0.062 | 0.054 | 0.126 | <2 ms |
| BM25 | 0.767 [0.70–0.84] | 0.986 | 0.936 | 0.958 | <1 ms |
| dense (Jina v3 + Chroma) | 0.558 | 0.906 | 0.512 | 0.896 | 0.8 s |
| hybrid, equal weights (before) | 0.753 | 0.951 | 0.742 | 0.972 | 0.8 s |
| hybrid, tuned weights | 0.782 | 0.978 | 0.897 | 0.968 | 0.8 s |
| **hybrid + Jina rerank (production)** | **0.817 [0.75–0.88]** | **0.980** | **0.957** | **1.000** | 1.5–1.7 s |

Known-item Hit@1 for the production system: Set B 0.961, Set C 0.904, Set D 1.000. Text-to-image (jina-clip-v2, 109 queries over 118 images): Recall@1 0.826, Recall@5 0.982, MRR 0.899. Full per-metric tables are in `eval/results.md`.

Paired deltas, production minus BM25 (nDCG@10): Set A +0.051 (CI 0.001 to 0.103), Set B −0.006 (−0.018 to 0.002), Set C +0.021 (−0.010 to 0.054), Set D +0.042 (0.000 to 0.086).

### Agent routing (30 messages, router `gpt-oss-20b`)

| Tool accuracy | Argument accuracy |
|---|---|
| 100% (30/30) | 100% (20/20) |

The first run scored 90% on arguments. The router set `location="beach"` for *"near the beach"* and ignored an explicit *"$"*. The `recommend_by_vibe` docstring now says what counts as a location filter and how `$` maps to `max_price`. That fix was made against these same 30 messages, so 100% is an optimistic figure. A held-out routing set is still to do.

### Answer quality (20 in-corpus + 10 out-of-corpus, generator and judge `gpt-oss-120b`)

| Metric | First run | After prompt fix |
|---|---|---|
| Faithfulness, mean (1–5) | 4.53 | **5.00** |
| Fully faithful answers | 57% | **100%** |
| Answers with judge-flagged unsupported claims | 13 / 30 | **0 / 30** |
| Answers with hallucinated restaurant names | 0 / 30 | 0 / 30 |
| Valid `[n]` citations | 100% | 100% |
| In-corpus answers that cite sources | 100% | 95% |
| Answer relevance, in-corpus (1–5) | 5.00 | 4.95 |
| Out-of-corpus requests correctly declined | 90% | 80% |
| In-corpus requests wrongly declined | 5% | 10% |

The first run never invented a restaurant. Its faithfulness losses were small embellishments: adjectives or dish preparations missing from the context, often the user's own wording repeated back as a fact about the restaurant. The answer prompt now forbids that. The trade-off is a more cautious model: a few more answers say "not mentioned". (Citation counts treat `【n】`, which gpt-oss sometimes writes, as `[n]`; the app normalizes it the same way.) With 10 out-of-corpus queries, 90% → 80% is a single query.

### Is it good enough?

| Target | Result | Met? |
|---|---|---|
| Production retrieval ≥ BM25 on every set | better on A and D; equal within noise on B and C | yes |
| Rerank latency p50 < 1.5 s end-to-end | 1.48–1.70 s (was 17.6 s with the LLM reranker) | borderline |
| Faithfulness ≥ 4.5 and ≥ 85% fully faithful | 5.00, 100% | yes |
| No hallucinated restaurant names | 0 / 30 | yes |
| Tool-routing accuracy ≥ 90% | 100% (tuned on the test set, see above) | yes, with caveat |
| Out-of-corpus requests declined ≥ 80% | 80% | just |

### What the numbers say

- **BM25 is still the strongest single retriever on this corpus.** The records are short and full of distinctive words, and Sets B and C are written from those records. Tuning on the dev split picked BM25 alone as the best setting. The chosen weights keep dense retrieval, which costs about 0.04 nDCG@10 on Set C (significant). In return, dense covers queries that share no words with the record. On the hand-written Set D, hybrid and BM25 are level.
- **Tuning the fusion weights fixed most of the vague-query gap.** On Set C, the tuned hybrid beats the equal-weight hybrid by +0.156 nDCG@10 (CI 0.110 to 0.210).
- **The cross-encoder reranker is where the quality comes from.** It lifts Set C Hit@1 from 0.789 (tuned hybrid) to 0.904 and gives a perfect score on Set D. It is about 10× faster than the LLM reranker, and it does not use Groq quota.
- **Generation is grounded but cautious.** No invented restaurants in either run. The remaining weak spot is judgement at the edges: about 1 in 10 answers declines a request the data could serve, and about 1 in 5 out-of-corpus requests still gets a "closest match" instead of a clear "not available".

### Not yet measured / next steps

- Held-out agent-routing set (the current one was used to fix the docstring).
- Larger answer-quality sample with confidence intervals; 30 answers is enough to catch big problems, not to separate close variants.
- Multi-turn conversations (follow-ups such as *"cheaper than that?"*).
- Chunking ablations: record and prose merged into one vector, and per-field vectors.
- `hybrid + LLM rerank` on the new splits (`python src/evaluate_retrieval.py --llm-rerank`), for comparison with the Jina reranker.

## Data Notes

- `structured_restaurant_data.json`: 210 restaurant records (`name`, `location`, `type`, `food_style`, `rating`, `price_range`, `signatures`, `vibe`, `environment`, `shortcomings`, `itemId`). Only 175 names are unique, because some chains appear in several cities; `itemId` tells them apart.
- `California-Culinary-Map.txt`: one prose paragraph per restaurant, in the same order as the JSON records. The index verifies each link with a name check.
- `augmented_user_review.json`: 10 reviews, linked by `itemId`, with photo URLs and captions.
- `augmented_food_recipe.json` + `synthetic_recipe_images/`: 109 recipes with photos, searchable through `search_images`.
