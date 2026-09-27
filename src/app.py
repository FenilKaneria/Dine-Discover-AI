# Libraries to create our MCP host application
import os
import sys
import json
import re
import asyncio
import time
import traceback
from pathlib import Path

import gradio as gr
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

# config loads .env and defines the Groq models and data paths
sys.path.insert(0, str(Path(__file__).parent))
from config import CHAT_MODEL, REASONING_MODEL, ROOT_DIR, answer_model, chat_model, llm_config_problem  # noqa: E402

# Configuration
SERVER_SCRIPT = str(Path(__file__).parent / "server.py")
MAX_TOOL_ROUNDS = 6  # Safety cap on the ReAct loop so a confused model can't spin forever
SYSTEM_PROMPT = """You are Dine-Discover-AI, an expert AI guide to California's vibrant restaurant scene.
Your job is to help users discover restaurants based on their preferences, vibes, and specific inquiries.
You have retrieval tools backed by a vector database (RAG):
1. get_restaurant_info — structured details for a restaurant by name.
2. recommend_by_vibe — hybrid semantic search for a natural-language request, with optional
   location / max_price (1-4) / min_rating filters. Pass the user's full request as `vibe`.
3. get_review — the user review (text, rating, photo descriptions) for a restaurant.
4. search_knowledge_base — semantic search over restaurant descriptions and reviews for open questions.
5. search_images — find food or restaurant photos matching a description.

Grounding rules:
- Always call a tool before recommending or describing a restaurant.
- Do not ask clarifying questions when a search is possible: search first with what the
  user gave you, then offer to narrow down (e.g. by city or budget).
- Use ONLY facts present in tool results. Never invent restaurants, dishes, prices or ratings.
- Name each restaurant you mention together with its location.
- If the tools return nothing relevant, say so and suggest a different search.
Be warm, concise and conversational.
"""

# Tools whose results contain image paths/URLs to show in the chat.
IMAGE_TOOLS = {"search_images", "get_review"}

# Launch the MCP server with the *same* interpreter running this app, so a venv
# or conda environment is inherited correctly (plain "python" may not be).
SERVER_PARAMS = StdioServerParameters(
    command=sys.executable,
    args=[SERVER_SCRIPT],
    env=os.environ.copy(),
)


# Startup checks — fail loudly here rather than mid-demo
def config_problem() -> str | None:
    """Return a human-readable problem with the configuration, or None if it looks OK."""
    problem = llm_config_problem()
    if problem:
        return problem
    if not Path(SERVER_SCRIPT).exists():
        return f"MCP server script not found at {SERVER_SCRIPT}."
    if not (ROOT_DIR / "data" / "chroma").exists():
        return "Vector index not found. Run `python src/build_index.py` first."
    return None


def tool_result_to_text(result) -> str:
    """Flatten an MCP CallToolResult into plain text for the LLM."""
    blocks = getattr(result, "content", None) or []
    parts = [getattr(block, "text", None) or str(block) for block in blocks]
    text = "\n".join(p for p in parts if p)
    return text or "(no result)"


def collect_images(tool_name: str, tool_output: str) -> list[str]:
    """Pull image paths/URLs out of a tool's JSON result."""
    if tool_name not in IMAGE_TOOLS:
        return []
    try:
        data = json.loads(tool_output)
    except (TypeError, ValueError):
        return []
    if tool_name == "search_images":
        return [img["uri"] for img in data.get("images", []) if img.get("uri")]
    return list(data.get("image_urls", []))


def extract_sources(tool_name: str, tool_output: str) -> tuple[list[dict], str | None]:
    """Pull the retrieved items (and the retrieval method, if reported) out of a
    tool's JSON result, for the trace panel and the numbered citation list."""
    try:
        data = json.loads(tool_output)
    except (TypeError, ValueError):
        return [], None
    if not isinstance(data, dict):
        return [], None

    sources: list[dict] = []
    if tool_name in ("recommend_by_vibe", "get_restaurant_info"):
        for r in data.get("results", []):
            sources.append({"name": r.get("name"), "location": r.get("location"), "detail": f"rating {r.get('rating')}"})
    elif tool_name == "search_knowledge_base":
        for p in data.get("passages", []):
            sources.append({"name": p.get("restaurant"), "location": None,
                            "detail": f"{p.get('source')}, sim {p.get('similarity')}"})
    elif tool_name == "get_review" and data.get("status") == "found":
        sources.append({"name": data.get("restaurant"), "location": data.get("location"),
                        "detail": f"review, {data.get('rating')}/5"})
    elif tool_name == "search_images":
        for img in data.get("images", []):
            sources.append({"name": img.get("name"), "location": None,
                            "detail": f"image, sim {img.get('similarity')}"})
    return [s for s in sources if s.get("name")], data.get("retrieval")


def numbered_sources(trace: list[dict]) -> list[dict]:
    """Deduplicated sources across all tool calls; list position + 1 is the citation number."""
    seen, out = set(), []
    for step in trace:
        for src in step["sources"]:
            key = (src["name"], src.get("location"))
            if key not in seen:
                seen.add(key)
                out.append(src)
    return out


def source_label(src: dict) -> str:
    return f"{src['name']} — {src['location']}" if src.get("location") else str(src["name"])


def render_trace(trace: list[dict]) -> str:
    """Markdown view of how the answer was built: MCP calls, retrieval, sources."""
    header = f"**MCP server:** Dine-Discover-AI · **router:** `{CHAT_MODEL}` · **generator:** `{REASONING_MODEL}`\n\n"
    if not trace:
        return header + "_No retrieval — answered directly._"

    rows = ["| # | MCP tool | Arguments | Latency | Hits |", "|---|---|---|---|---|"]
    for i, step in enumerate(trace, start=1):
        args = json.dumps(step["args"], ensure_ascii=False).replace("|", "\\|")
        rows.append(f"| {i} | `{step['tool']}` | `{args}` | {step['ms']:.0f} ms | {len(step['sources'])} |")

    methods = sorted({step["retrieval"] for step in trace if step.get("retrieval")})
    lines = [header, "\n".join(rows)]
    if methods:
        lines.append("\n**Retrieval:** " + "; ".join(methods))

    sources = numbered_sources(trace)
    if sources:
        lines.append("\n**Sources used as context:**")
        lines += [
            f"{n}. {source_label(s)}" + (f" ({s['detail']})" if s.get("detail") is not None else "")
            for n, s in enumerate(sources, start=1)
        ]
    return "\n".join(lines)


def describe_exception(exc: BaseException) -> str:
    """anyio wraps anything raised inside the MCP session in an ExceptionGroup.
    Unwrap it so the chat shows the real cause (e.g. AuthenticationError)."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return f"{type(exc).__name__}: {exc}"


def response_to_text(response) -> str:
    """Normalize a LangChain message's content into a plain string."""
    raw = response.content
    if isinstance(raw, list):
        return " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in raw
        ).strip()
    return str(raw).strip()


# MCP Host — ReAct Agent Loop
async def chat_with_agent(user_message: str, history: list) -> tuple[str, list[str], list[dict]]:
    """Connect to the MCP server, discover tools, and run a ReAct loop.

    The fast model (gpt-oss-20b) decides which retrieval tools to call. Once it
    has the context, the stronger model (gpt-oss-120b) writes the final,
    grounded answer. Returns (answer, image paths/URLs to display, trace of
    every MCP tool call with its latency and retrieved sources)."""
    images: list[str] = []
    trace: list[dict] = []
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Discover available tools from the MCP server
            tools_result = await session.list_tools()

            # Convert MCP tool schemas to OpenAI-style tool definitions for the LLM
            openai_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description or "",
                        "parameters": t.inputSchema,
                    },
                }
                for t in tools_result.tools
            ]

            model = chat_model(CHAT_MODEL).bind_tools(openai_tools)
            used_tools = False

            # Build the message list from chat history and the new user message
            messages = [SystemMessage(content=SYSTEM_PROMPT)]
            for msg in history:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user" and content:
                    messages.append(HumanMessage(content=content))
                elif role == "assistant" and content:
                    messages.append(AIMessage(content=content))
            messages.append(HumanMessage(content=user_message))

            # ReAct loop — call tools until the LLM returns a plain text reply
            for _ in range(MAX_TOOL_ROUNDS):
                response = await model.ainvoke(messages)
                messages.append(response)

                # No tool calls means retrieval is finished
                if not response.tool_calls:
                    draft = response_to_text(response)
                    if not used_tools:
                        return draft or "(the model returned an empty reply)", images, trace
                    # Generation step: the stronger model answers from the retrieved context.
                    answer = await synthesize_answer(messages, user_message, trace)
                    return answer or draft or "(the model returned an empty reply)", images, trace

                used_tools = True
                # Execute each tool call via the MCP server and feed results back.
                # A failing tool must come back as a ToolMessage, not an exception,
                # otherwise the conversation is left with an unanswered tool call.
                for tool_call in response.tool_calls:
                    started = time.perf_counter()
                    try:
                        result = await session.call_tool(
                            tool_call["name"], arguments=tool_call["args"] or {}
                        )
                        tool_output = tool_result_to_text(result)
                        images += collect_images(tool_call["name"], tool_output)
                    except Exception as exc:
                        tool_output = f"Tool '{tool_call['name']}' failed: {exc}"
                    sources, retrieval = extract_sources(tool_call["name"], tool_output)
                    trace.append({
                        "tool": tool_call["name"],
                        "args": tool_call["args"] or {},
                        "ms": (time.perf_counter() - started) * 1000,
                        "sources": sources,
                        "retrieval": retrieval,
                    })
                    messages.append(
                        ToolMessage(content=tool_output, tool_call_id=tool_call["id"])
                    )

            # Loop cap hit — ask the model once more for a plain answer
            messages.append(
                HumanMessage(content="Please answer now using what you already found, without calling more tools.")
            )
            answer = await synthesize_answer(messages, user_message, trace)
            return answer or "I wasn't able to complete that request. Please try again.", images, trace


ANSWER_PROMPT = """Answer the user's latest question using ONLY the retrieved context below.
Describe each restaurant only with facts and wording found in the context: do not add
adjectives, dish preparations or vibes that the context does not state, and never apply the
user's own words to a restaurant unless the context says the same. If a detail the user asked
for is not in the context, say it is not mentioned.
Name each restaurant you mention with its location. If nothing matches exactly, present
the closest matches and say how they differ; only if the context is empty or unrelated,
say so and suggest a different search. Do not mention tools, JSON or file paths
(photos are displayed to the user automatically). When the context starts with a numbered
source list, cite each restaurant you mention with its number, e.g. "Iron & Embers [2]".

Conversation so far:
{conversation}

Retrieved context:
{context}

User question: {question}"""


def grounded_context(tool_outputs: list[str], trace: list[dict]) -> str:
    """Tool results joined into the generation context, headed by the numbered
    source list the answer cites as [n]. Shared with evaluate_answers.py."""
    context = "\n\n".join(tool_outputs)
    sources = numbered_sources(trace)
    if sources and context:
        listing = "\n".join(f"[{n}] {source_label(s)}" for n, s in enumerate(sources, start=1))
        context = f"Sources:\n{listing}\n\n{context}"
    return context


async def synthesize_answer(messages: list, question: str, trace: list[dict] | None = None) -> str:
    """RAG generation step: retrieved context + question -> grounded answer.

    The context is passed as plain text (not as tool messages) so the reasoning
    model answers directly instead of trying to call tools itself. A numbered
    source list is prepended so the answer can cite [n], matching the trace panel."""
    context = grounded_context([m.content for m in messages if isinstance(m, ToolMessage)], trace or [])
    conversation = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in messages[1:-1]
        if isinstance(m, (HumanMessage, AIMessage)) and isinstance(m.content, str) and m.content
    )
    prompt = ANSWER_PROMPT.format(
        conversation=conversation or "(none)", context=context or "(nothing retrieved)", question=question
    )
    final = await answer_model().ainvoke(
        [SystemMessage(content=SYSTEM_PROMPT.split("You have retrieval tools")[0]), HumanMessage(content=prompt)]
    )
    # gpt-oss sometimes writes citations as 【1】; normalize to [1] for the UI and eval.
    return re.sub(r"【(\d+)】", r"[\1]", response_to_text(final))


def text_history(history: list) -> list:
    """Chat history minus image messages (the LLM only needs the text)."""
    return [m for m in history or [] if isinstance(m.get("content"), str)]


# Gradio Event Handler
async def handle_chat(user_message, history):
    if history is None:
        history = []
    if not user_message or not user_message.strip():
        yield history, gr.update()
        return

    prior_history = text_history(history)

    # Show a thinking placeholder while the agent runs
    history = list(history) + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": "Thinking..."},
    ]
    yield history, "_Retrieving..._"

    problem = config_problem()
    if problem:
        history[-1] = {"role": "assistant", "content": f"**Configuration problem:** {problem}"}
        yield history, f"_Not run: {problem}_"
        return

    # Any failure (bad API key, server crash, network) becomes a chat message
    # instead of a traceback that kills the UI mid-demo.
    images: list[str] = []
    try:
        response_text, images, trace = await chat_with_agent(user_message, prior_history)
        trace_text = render_trace(trace)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        traceback.print_exc()
        response_text = (
            f"**Something went wrong:** `{describe_exception(exc)}`\n\n"
            "Check that GROQ_API_KEY / JINA_API_KEY in `.env` are valid and that the "
            "MCP server can start (`python src/client.py` tests it without the LLM)."
        )
        trace_text = f"_Failed: `{describe_exception(exc)}`_"

    history[-1] = {"role": "assistant", "content": response_text}
    # Show retrieved photos under the answer (local recipe files or review URLs).
    for image in dict.fromkeys(images):
        history.append({"role": "assistant", "content": {"path": image}})
    yield history, trace_text


def find_similar_dishes(image_path):
    """Image-to-image search: embed the uploaded photo with jina-clip-v2."""
    if not image_path:
        return []
    from embeddings import image_to_base64
    from retrieval import get_retriever

    hits = get_retriever().search_images(image_b64=image_to_base64(Path(image_path)), k=6)
    return [
        (str(ROOT_DIR / h["uri"]) if h["source"] == "recipe" else h["uri"], f"{h['name']} ({h['similarity']})")
        for h in hits
    ]


EXAMPLES = [
    "Find me some moody restaurants",
    "Tell me about Iron & Embers",
    "Cheap tacos near the beach under $$",
    "Show me photos of a margherita pizza",
]


# Gradio Interface
with gr.Blocks(title="Dine-Discover-AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Dine-Discover-AI\nYour AI guide to California's restaurant scene. "
        "Ask about restaurants by name, cuisine, vibe or budget — answers are grounded in a "
        f"RAG index (Jina embeddings + ChromaDB, hybrid search) and written by {REASONING_MODEL}."
    )

    # type="messages" is required — this app yields OpenAI-style {role, content} dicts
    # No LaTeX: "$" is a price symbol here, and "$ ... $" pairs would render as math.
    chatbot = gr.Chatbot(height=500, type="messages", latex_delimiters=[])
    msg_input = gr.Textbox(
        label="Ask about restaurants",
        placeholder='e.g., "Find me a moody spot in DTLA" or "Tell me about Sakura Garden"',
    )

    with gr.Accordion("How this answer was built (RAG trace)", open=False):
        trace_md = gr.Markdown("_Ask something to see which MCP tools were called and what was retrieved._")

    with gr.Row():
        example_buttons = [gr.Button(text, size="sm") for text in EXAMPLES]

    with gr.Accordion("Find similar dishes from a photo (multimodal search)", open=False):
        with gr.Row():
            photo_input = gr.Image(type="filepath", label="Upload a food photo", height=260)
            similar_gallery = gr.Gallery(label="Most similar dishes", columns=3, height=260)
        photo_input.change(find_similar_dishes, photo_input, similar_gallery)

    # Run the agent, then clear the textbox (clearing in a separate listener would
    # race the handler and could blank the input before it is read).
    msg_input.submit(handle_chat, [msg_input, chatbot], [chatbot, trace_md]).then(
        lambda: "", None, msg_input
    )

    for button, text in zip(example_buttons, EXAMPLES):
        button.click(lambda t=text: t, None, msg_input).then(
            handle_chat, [msg_input, chatbot], [chatbot, trace_md]
        ).then(lambda: "", None, msg_input)

# Launch the App
if __name__ == "__main__":
    print("Starting Dine-Discover-AI...")

    problem = config_problem()
    if problem:
        print(f"\n  WARNING: {problem}")
        print("  The UI will still start, but chat responses will show this error.\n")

    # share=True needs to download a tunnel binary and reach Gradio's servers,
    # which often fails on locked-down networks. Opt in with SHARE=true.
    demo.queue().launch(
        share=os.environ.get("SHARE", "").lower() in ("1", "true", "yes"),
        server_name=os.environ.get("SERVER_NAME", "127.0.0.1"),
        show_error=True,
        # Recipe photos live under ./data and are served to the chat/gallery.
        allowed_paths=[str(ROOT_DIR / "data" / "raw")],
    )
