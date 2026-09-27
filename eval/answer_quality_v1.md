# Answer quality (20 in-corpus + 10 out-of-corpus queries)

Generator and judge: `openai/gpt-oss-120b`.

| Metric | Value |
|---|---|
| Faithfulness, mean (1-5) | 4.53 |
| Fully faithful (score 5) | 57% |
| Answer relevance, in-corpus (1-5) | 5 |
| Out-of-corpus correctly declined | 90% |
| In-corpus wrongly declined | 5% |
| Answers with hallucinated restaurant names | 0 / 30 |
| Answers with judge-flagged unsupported claims | 13 / 30 |
| Valid citations | 100% |
| In-corpus answers with citations | 100% |
