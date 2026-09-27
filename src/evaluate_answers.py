"""Answer-quality evaluation of the full RAG pipeline.

For each query: retrieve with the production tool (`recommend_by_vibe`: hybrid
retrieval + rerank) -> build the context exactly as the app does (numbered
source list + tool output) -> generate with the production prompt and model
(gpt-oss-120b) -> score.

Query mix
  Set A  rule-labelled requests (eval set shared with evaluate_retrieval.py)
  Set D  hand-written human-style requests (eval/queries_d.json)
  OOC    out-of-corpus requests the data cannot answer; the right answer is
         to say so instead of recommending something

LLM judge (gpt-oss-120b, separate call)
  faithfulness      1-5  every factual claim is supported by the retrieved context
  answer_relevance  1-5  the answer addresses the user's request
  declined          the answer says nothing suitable was found (scored on OOC)
  unsupported_claims     claims not found in the context

Deterministic checks (no LLM)
  hallucinated_names  corpus restaurant names in the answer that are NOT in the
                      retrieved context
  citations           [n] markers; valid when n is in the source list
  cited_answers       share of in-corpus answers with at least one citation

Results: eval/answer_quality.json and eval/answer_quality.md

Usage:
    python src/evaluate_answers.py                    # 12 A + 8 D + 10 OOC
    python src/evaluate_answers.py --a 5 --d 5 --ooc 5
"""
import argparse
import json
import re
import statistics
import sys
import time

from app import ANSWER_PROMPT, SYSTEM_PROMPT, extract_sources, grounded_context, numbered_sources
from config import EVAL_DIR, REASONING_MODEL, RESTAURANT_DATA_PATH, groq_client
from evaluate_retrieval import SET_A
from server import recommend_by_vibe

JUDGE_PROMPT = """You are a strict evaluator of a retrieval-augmented restaurant assistant.

User request: {query}

Retrieved context (the ONLY allowed source of facts):
{context}

Assistant answer:
{answer}

Score the answer:
- faithfulness (1-5): 5 = every factual claim (names, locations, dishes, prices, ratings, vibes)
  is supported by the context; 1 = mostly unsupported or invented.
- answer_relevance (1-5): 5 = directly and fully addresses the request; 1 = off-topic.
- declined (true/false): true if the answer clearly tells the user that nothing matching
  the request was found, rather than presenting restaurants as a match.
List any unsupported claims verbatim.

Return JSON only: {{"faithfulness": <int>, "answer_relevance": <int>, "declined": <bool>, "unsupported_claims": [<str>], "reason": "<one sentence>"}}"""


def chat(messages: list[dict], json_mode: bool = False) -> str:
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    for attempt in range(4):
        try:
            resp = groq_client().chat.completions.create(
                model=REASONING_MODEL, messages=messages, temperature=0, **kwargs
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            print(f"    retry {attempt + 1}: {type(exc).__name__}: {str(exc)[:120]}")
            time.sleep(30)
    raise RuntimeError("LLM call failed repeatedly")


def corpus_names() -> list[str]:
    """Distinctive restaurant names (short generic ones would match ordinary words)."""
    records = json.loads(RESTAURANT_DATA_PATH.read_text(encoding="utf-8"))
    names = {str(r.get("name", "")).strip() for r in records}
    return sorted((n for n in names if len(n) >= 6), key=len, reverse=True)


def hallucinated_names(answer: str, context: str, names: list[str]) -> list[str]:
    answer_l, context_l = answer.lower(), context.lower()
    found = []
    for name in names:  # longest first, so "The Velvet Vine" is not also counted as "Velvet Vine"
        pattern = r"(?<!\w)" + re.escape(name.lower()) + r"(?!\w)"
        if re.search(pattern, answer_l) and not any(name.lower() in f.lower() for f in found):
            if name.lower() not in context_l:
                found.append(name)
    return found


def citation_check(answer: str, n_sources: int) -> tuple[int, int]:
    """(total [n] citations, valid ones)."""
    answer = re.sub(r"【(\d+)】", r"[\1]", answer)  # same normalization as the app
    cited = [int(n) for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", answer) for n in re.split(r"\s*,\s*", group)]
    return len(cited), sum(1 for n in cited if 1 <= n <= n_sources)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=int, default=12, help="Set A queries")
    parser.add_argument("--d", type=int, default=8, help="Set D known-item queries")
    parser.add_argument("--ooc", type=int, default=10, help="out-of-corpus queries")
    args = parser.parse_args()

    set_d = json.loads((EVAL_DIR / "queries_d.json").read_text(encoding="utf-8"))
    queries = (
        [("A", q) for q, *_ in SET_A[: args.a]]
        + [("D", q["query"]) for q in set_d["known_item"][: args.d]]
        + [("OOC", q) for q in set_d["out_of_corpus"][: args.ooc]]
    )
    names = corpus_names()

    rows = []
    for kind, query in queries:
        tool_output = recommend_by_vibe(query)
        sources, retrieval = extract_sources("recommend_by_vibe", tool_output)
        trace = [{"tool": "recommend_by_vibe", "args": {"vibe": query}, "ms": 0, "sources": sources, "retrieval": retrieval}]
        context = grounded_context([tool_output], trace)
        prompt = ANSWER_PROMPT.format(conversation="(none)", context=context, question=query)
        answer = chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT.split("You have retrieval tools")[0]},
                {"role": "user", "content": prompt},
            ]
        )
        verdict = json.loads(
            chat(
                [{"role": "user", "content": JUDGE_PROMPT.format(query=query, context=context, answer=answer)}],
                json_mode=True,
            )
        )
        n_cites, valid_cites = citation_check(answer, len(numbered_sources(trace)))
        row = {
            "set": kind,
            "query": query,
            "answer": answer,
            **verdict,
            "hallucinated_names": hallucinated_names(answer, context, names),
            "citations": n_cites,
            "valid_citations": valid_cites,
        }
        rows.append(row)
        print(
            f"  [{kind:3s}] {query[:45]:45s} faith={row.get('faithfulness')} rel={row.get('answer_relevance')} "
            f"declined={row.get('declined')} cites={valid_cites}/{n_cites} halluc={row['hallucinated_names']}",
            flush=True,
        )

    in_corpus = [r for r in rows if r["set"] != "OOC"]
    ooc = [r for r in rows if r["set"] == "OOC"]
    faith = [r["faithfulness"] for r in rows]
    total_cites = sum(r["citations"] for r in rows)
    summary = {
        "n_queries": len(rows),
        "n_in_corpus": len(in_corpus),
        "n_out_of_corpus": len(ooc),
        "generator": REASONING_MODEL,
        "judge": REASONING_MODEL,
        "faithfulness_mean": round(statistics.mean(faith), 2),
        "fully_faithful_rate": round(sum(1 for f in faith if f == 5) / len(faith), 3),
        "answer_relevance_mean_in_corpus": round(statistics.mean(r["answer_relevance"] for r in in_corpus), 2)
        if in_corpus else None,
        "ooc_decline_rate": round(sum(1 for r in ooc if r.get("declined")) / len(ooc), 3) if ooc else None,
        "false_decline_rate_in_corpus": round(sum(1 for r in in_corpus if r.get("declined")) / len(in_corpus), 3)
        if in_corpus else None,
        "answers_with_unsupported_claims": sum(1 for r in rows if r.get("unsupported_claims")),
        "answers_with_hallucinated_names": sum(1 for r in rows if r["hallucinated_names"]),
        "citation_validity": round(sum(r["valid_citations"] for r in rows) / total_cites, 3) if total_cites else None,
        "cited_answers_in_corpus": round(sum(1 for r in in_corpus if r["citations"]) / len(in_corpus), 3)
        if in_corpus else None,
    }
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / "answer_quality.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    def pct(v):
        return "-" if v is None else f"{v:.0%}"

    md = (
        f"# Answer quality ({summary['n_in_corpus']} in-corpus + {summary['n_out_of_corpus']} out-of-corpus queries)\n\n"
        f"Generator and judge: `{REASONING_MODEL}`.\n\n"
        "| Metric | Value |\n|---|---|\n"
        f"| Faithfulness, mean (1-5) | {summary['faithfulness_mean']} |\n"
        f"| Fully faithful (score 5) | {pct(summary['fully_faithful_rate'])} |\n"
        f"| Answer relevance, in-corpus (1-5) | {summary['answer_relevance_mean_in_corpus']} |\n"
        f"| Out-of-corpus correctly declined | {pct(summary['ooc_decline_rate'])} |\n"
        f"| In-corpus wrongly declined | {pct(summary['false_decline_rate_in_corpus'])} |\n"
        f"| Answers with hallucinated restaurant names | {summary['answers_with_hallucinated_names']} / {summary['n_queries']} |\n"
        f"| Answers with judge-flagged unsupported claims | {summary['answers_with_unsupported_claims']} / {summary['n_queries']} |\n"
        f"| Valid citations | {pct(summary['citation_validity'])} |\n"
        f"| In-corpus answers with citations | {pct(summary['cited_answers_in_corpus'])} |\n"
    )
    (EVAL_DIR / "answer_quality.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
