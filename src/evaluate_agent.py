"""Agent routing evaluation: does the router model pick the right MCP tool and
extract the right arguments?

For each message in eval/queries_agent.json the tools are discovered from the
real MCP server (exactly as the app does), bound to the router model
(CHAT_MODEL), and the model's FIRST response is scored. Tools are not executed.

  tool_accuracy     first tool call (or no call) is one of the acceptable tools
  arg_accuracy      on items with expected arguments: every expected argument
                    matches, and no unrequested filter was added
                    (recommend_by_vibe: location / max_price / min_rating;
                    name tools: restaurant_name contains the expected name)

Results: eval/agent_routing.json and eval/agent_routing.md

Usage:
    python src/evaluate_agent.py
"""
import asyncio
import json
import sys
import time
from collections import defaultdict

from langchain_core.messages import HumanMessage, SystemMessage
from mcp import ClientSession
from mcp.client.stdio import stdio_client

from app import SERVER_PARAMS, SYSTEM_PROMPT
from config import CHAT_MODEL, EVAL_DIR, chat_model

FILTERS = ("location", "max_price", "min_rating")


def args_match(tool: str, got: dict, expected: dict) -> tuple[bool, str]:
    """Compare extracted arguments with the expected ones. Returns (ok, reason)."""
    if tool in ("get_restaurant_info", "get_review"):
        name = str(got.get("restaurant_name", "")).lower()
        want = expected["restaurant_name"].lower()
        return (want in name, "" if want in name else f"restaurant_name={got.get('restaurant_name')!r}")
    if tool == "recommend_by_vibe":
        problems = []
        for key in FILTERS:
            value, want = got.get(key), expected.get(key)
            if want is None:
                if value not in (None, "", 0):
                    problems.append(f"unrequested {key}={value!r}")
            elif key == "location":
                if not value or str(want).lower() not in str(value).lower():
                    problems.append(f"location={value!r} (want {want!r})")
            elif value is None or float(value) != float(want):
                problems.append(f"{key}={value!r} (want {want!r})")
        return (not problems, "; ".join(problems))
    return True, ""


async def route(model, query: str):
    for attempt in range(4):
        try:
            return await model.ainvoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=query)])
        except Exception as exc:
            print(f"    retry {attempt + 1}: {type(exc).__name__}: {str(exc)[:120]}")
            time.sleep(30)
    raise RuntimeError("router model failed repeatedly")


async def main_async() -> int:
    items = json.loads((EVAL_DIR / "queries_agent.json").read_text(encoding="utf-8"))["items"]

    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
    openai_tools = [
        {"type": "function", "function": {"name": t.name, "description": t.description or "", "parameters": t.inputSchema}}
        for t in tools
    ]
    model = chat_model(CHAT_MODEL, temperature=0).bind_tools(openai_tools)

    rows = []
    for item in items:
        response = await route(model, item["query"])
        call = response.tool_calls[0] if response.tool_calls else None
        tool = call["name"] if call else "none"
        got_args = (call or {}).get("args") or {}
        tool_ok = tool in item["tools"]
        needs_args = bool(item["args"]) or tool == "recommend_by_vibe"
        args_ok, reason = args_match(tool, got_args, item["args"]) if tool_ok and needs_args else (tool_ok, "")
        rows.append(
            {
                "query": item["query"],
                "expected_tools": item["tools"],
                "tool": tool,
                "args": got_args,
                "tool_ok": tool_ok,
                "args_checked": needs_args,
                "args_ok": args_ok if needs_args else None,
                "reason": reason if tool_ok else f"called {tool}",
            }
        )
        mark = "ok " if tool_ok and (args_ok or not needs_args) else "BAD"
        print(f"  {mark} {item['query'][:50]:50s} -> {tool} {json.dumps(got_args)[:80]} {reason}", flush=True)
        time.sleep(2)  # stay under the Groq free-tier tokens-per-minute limit

    by_tool = defaultdict(lambda: [0, 0])
    for r in rows:
        key = r["expected_tools"][0]
        by_tool[key][0] += r["tool_ok"]
        by_tool[key][1] += 1
    checked = [r for r in rows if r["args_checked"]]
    summary = {
        "router_model": CHAT_MODEL,
        "n_queries": len(rows),
        "tool_accuracy": round(sum(r["tool_ok"] for r in rows) / len(rows), 3),
        "arg_accuracy": round(sum(bool(r["args_ok"]) for r in checked) / len(checked), 3) if checked else None,
        "n_arg_checked": len(checked),
        "per_expected_tool": {k: f"{ok}/{n}" for k, (ok, n) in sorted(by_tool.items())},
    }
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / "agent_routing.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    failures = [r for r in rows if not r["tool_ok"] or r["args_ok"] is False]
    md = [
        f"# Agent routing ({summary['n_queries']} queries, router `{CHAT_MODEL}`)\n",
        "| Tool accuracy | Argument accuracy | Per expected tool |",
        "|---|---|---|",
        f"| {summary['tool_accuracy']:.0%} | {summary['arg_accuracy']:.0%} (n={len(checked)}) | "
        + ", ".join(f"{k} {v}" for k, v in summary["per_expected_tool"].items())
        + " |",
    ]
    if failures:
        md.append("\n**Failures**\n")
        md += [f"- \"{r['query']}\" → `{r['tool']}` {json.dumps(r['args'])}: {r['reason']}" for r in failures]
    (EVAL_DIR / "agent_routing.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # tables use non-ASCII (Δ, –)
    sys.exit(asyncio.run(main_async()))
