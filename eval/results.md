# Retrieval evaluation (2026-09-27)

Embeddings `jina-embeddings-v3` (text) / `jina-clip-v2` (images), ChromaDB, reranker `jina-reranker-v2-base-multilingual`, RRF weights [0.25, 0.25, 2.0], corpus 210 restaurants.

**Set A — rule-labelled requests** (40 queries; [95% bootstrap CI])

| System | recall@10 | precision@5 | mrr | ndcg@10 | latency p50 ms |
|---|---|---|---|---|---|
| keyword | 0.3043 | 0.305 | 0.4104 [0.2884–0.5342] | 0.3364 [0.2348–0.4389] | 1.0 |
| bm25 | 0.7129 | 0.695 | 0.8979 [0.8208–0.9708] | 0.7667 [0.696–0.8356] | 0.2 |
| dense | 0.4915 | 0.495 | 0.7382 [0.6132–0.8549] | 0.5577 [0.4537–0.6619] | 747.1 |
| hybrid (equal weights) | 0.6986 | 0.645 | 0.8662 [0.7667–0.9542] | 0.7533 [0.6719–0.826] | 745.8 |
| hybrid | 0.7294 | 0.71 | 0.8896 [0.8125–0.9583] | 0.782 [0.7125–0.8487] | 749.5 |
| hybrid+rerank | 0.7474 | 0.735 | 0.8875 [0.8083–0.9583] | 0.8172 [0.7471–0.8816] | 1482.5 |

- hybrid − bm25: Δndcg@10 = +0.0154 (95% CI -0.0126 to 0.0473)
- hybrid+rerank − bm25: Δndcg@10 = +0.0506 (95% CI 0.0005 to 0.1032)
- hybrid (tuned) − hybrid (equal weights): Δndcg@10 = +0.0287 (95% CI -0.0191 to 0.0778)

**Set B — known-item paraphrases (test split)** (103 queries; [95% bootstrap CI])

| System | hit@1 | hit@5 | mrr | ndcg@10 | latency p50 ms |
|---|---|---|---|---|---|
| keyword | 0.0097 | 0.0583 | 0.0377 [0.0161–0.0656] | 0.0615 [0.0316–0.0962] | 1.7 |
| bm25 | 0.9806 | 0.9903 | 0.9854 [0.9612–1.0] | 0.9859 [0.9621–1.0] | 0.3 |
| dense | 0.8155 | 0.9612 | 0.8835 [0.8285–0.9337] | 0.9057 [0.859–0.9471] | 774.0 |
| hybrid (equal weights) | 0.9126 | 0.9709 | 0.9409 [0.8981–0.9757] | 0.9509 [0.913–0.9801] | 886.8 |
| hybrid | 0.9612 | 0.9903 | 0.9741 [0.945–0.9951] | 0.9775 [0.9509–0.9964] | 856.9 |
| hybrid+rerank | 0.9612 | 0.9903 | 0.9757 [0.9466–0.9951] | 0.9795 [0.953–0.9964] | 1670.0 |

- hybrid − bm25: Δndcg@10 = -0.0084 (95% CI -0.0217 to 0.0)
- hybrid+rerank − bm25: Δndcg@10 = -0.0064 (95% CI -0.0179 to 0.0016)
- hybrid (tuned) − hybrid (equal weights): Δndcg@10 = +0.0266 (95% CI 0.0091 to 0.0492)

**Set C — vague known-item, no names/places/dishes (test split)** (104 queries; [95% bootstrap CI])

| System | hit@1 | hit@5 | mrr | ndcg@10 | latency p50 ms |
|---|---|---|---|---|---|
| keyword | 0.0096 | 0.0481 | 0.0348 [0.014–0.0603] | 0.0542 [0.0259–0.0864] | 1.7 |
| bm25 | 0.8942 | 0.9712 | 0.9218 [0.8737–0.9607] | 0.9362 [0.8942–0.969] | 0.3 |
| dense | 0.2981 | 0.6346 | 0.4387 [0.3599–0.5187] | 0.5124 [0.4374–0.5856] | 889.0 |
| hybrid (equal weights) | 0.5577 | 0.8846 | 0.6838 [0.6002–0.7561] | 0.7416 [0.6744–0.8027] | 900.4 |
| hybrid | 0.7885 | 0.9712 | 0.8661 [0.8088–0.9199] | 0.8971 [0.8555–0.9379] | 841.2 |
| hybrid+rerank | 0.9038 | 0.9904 | 0.9445 [0.9083–0.9736] | 0.9567 [0.9292–0.9798] | 1702.5 |

- hybrid − bm25: Δndcg@10 = -0.0391 (95% CI -0.0737 to -0.008)
- hybrid+rerank − bm25: Δndcg@10 = +0.0205 (95% CI -0.0098 to 0.0537)
- hybrid (tuned) − hybrid (equal weights): Δndcg@10 = +0.1555 (95% CI 0.1098 to 0.2102)

**Set D — hand-written human-style known-item** (25 queries; [95% bootstrap CI])

| System | hit@1 | hit@5 | mrr | ndcg@10 | latency p50 ms |
|---|---|---|---|---|---|
| keyword | 0.04 | 0.16 | 0.0914 [0.02–0.1886] | 0.1264 [0.0345–0.2275] | 1.8 |
| bm25 | 0.88 | 1.0 | 0.94 [0.88–1.0] | 0.9582 [0.9139–1.0] | 0.3 |
| dense | 0.84 | 0.92 | 0.8757 [0.7514–0.97] | 0.8958 [0.7877–0.9772] | 843.6 |
| hybrid (equal weights) | 0.96 | 0.96 | 0.964 [0.892–1.0] | 0.9716 [0.9147–1.0] | 888.7 |
| hybrid | 0.92 | 1.0 | 0.9533 [0.88–1.0] | 0.9677 [0.92–1.0] | 836.2 |
| hybrid+rerank | 1.0 | 1.0 | 1.0 [1.0–1.0] | 1.0 [1.0–1.0] | 1607.0 |

- hybrid − bm25: Δndcg@10 = +0.0095 (95% CI -0.0505 to 0.0591)
- hybrid+rerank − bm25: Δndcg@10 = +0.0418 (95% CI 0.0 to 0.0861)
- hybrid (tuned) − hybrid (equal weights): Δndcg@10 = -0.0038 (95% CI -0.0368 to 0.0215)

**Text-to-image** (109 recipe-name queries over 118 images)

| Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|
| 0.8257 | 0.9817 | 1.0 | 0.8985 |
