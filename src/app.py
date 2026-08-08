# Libraries to create our MCP host application
import os
import sys
import asyncio
import logging
import traceback
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

# Load .env from the project root before anything reads os.environ.
# Unparseable lines are skipped silently; config_problem() below reports the
# consequences in plain English instead of leaking parser warnings to the console.
logging.getLogger("dotenv.main").setLevel(logging.ERROR)
load_dotenv(Path(__file__).parent.parent / ".env")

# Configuration
SERVER_SCRIPT = str(Path(__file__).parent / "server.py")
MAX_TOOL_ROUNDS = 6  # Safety cap on the ReAct loop so a confused model can't spin forever
SYSTEM_PROMPT = """You are the Connoisseur Companion, an expert AI guide to California's vibrant restaurant scene.
Your job is to help users discover restaurants based on their preferences, vibes, and specific inquiries.
You have access to tools that can:
1. Search for specific restaurants by name to get their structured details.
2. Recommend restaurants based on atmospheric vibes (e.g., "moody", "romantic", "zen").
3. Retrieve detailed user reviews for specific restaurants.

Always use the provided tools to fetch accurate information before answering. Be enthusiastic, helpful, and conversational in your responses. If a user asks for something you can't find, let them know politely and suggest alternative searches.
"""

# Launch the MCP server with the *same* interpreter running this app, so a venv
# or conda environment is inherited correctly (plain "python" may not be).
SERVER_PARAMS = StdioServerParameters(
    command=sys.executable,
    args=[SERVER_SCRIPT],
    env=os.environ.copy(),
)


# Startup checks — fail loudly here rather than mid-demo
def config_problem() -> str | None:
    """Return a human-readable problem with the LLM config, or None if it looks OK."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    api_base = os.environ.get("OPENAI_API_BASE", "").strip()

    if not api_key:
        return (
            "OPENAI_API_KEY is not set. Add it to the .env file in the project root, "
            "e.g. OPENAI_API_KEY=sk-... (or point OPENAI_API_BASE at a local model server)."
        )
    # A real OpenAI key is far longer than this; a local server can use any dummy key.
    if not api_base and len(api_key) < 20:
        return (
            f"OPENAI_API_KEY looks invalid (only {len(api_key)} characters). "
            "Replace it in .env with a real key, or set OPENAI_API_BASE to a local "
            "model server such as http://localhost:11434/v1."
        )
    if not Path(SERVER_SCRIPT).exists():
        return f"MCP server script not found at {SERVER_SCRIPT}."
    return None


# Initializing the LLM via OpenAI API format
def make_model():
    return ChatOpenAI(
        model=os.environ.get("MODEL_NAME", "gpt-4o-mini"),
        temperature=0.7,
        timeout=60,
        max_retries=2,
        # OPENAI_API_KEY and OPENAI_API_BASE are automatically picked up by the SDK
    )


def tool_result_to_text(result) -> str:
    """Flatten an MCP CallToolResult into plain text for the LLM."""
    blocks = getattr(result, "content", None) or []
    parts = [getattr(block, "text", None) or str(block) for block in blocks]
    text = "\n".join(p for p in parts if p)
    return text or "(no result)"


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
async def chat_with_agent(user_message: str, history: list) -> str:
    """Connect to the MCP server, discover tools, and run a ReAct loop.
    The LLM decides which tools to call, calls them via the MCP server,
    and repeats until it produces a final text response."""
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

            model = make_model().bind_tools(openai_tools)

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

                # No tool calls means the LLM is done — return the final response
                if not response.tool_calls:
                    return response_to_text(response) or "(the model returned an empty reply)"

                # Execute each tool call via the MCP server and feed results back.
                # A failing tool must come back as a ToolMessage, not an exception,
                # otherwise the conversation is left with an unanswered tool call.
                for tool_call in response.tool_calls:
                    try:
                        result = await session.call_tool(
                            tool_call["name"], arguments=tool_call["args"] or {}
                        )
                        tool_output = tool_result_to_text(result)
                    except Exception as exc:
                        tool_output = f"Tool '{tool_call['name']}' failed: {exc}"
                    messages.append(
                        ToolMessage(content=tool_output, tool_call_id=tool_call["id"])
                    )

            # Loop cap hit — ask the model once more for a plain answer
            messages.append(
                HumanMessage(content="Please answer now using what you already found, without calling more tools.")
            )
            final = await make_model().ainvoke(messages)
            return response_to_text(final) or "I wasn't able to complete that request. Please try again."


# Gradio Event Handler
async def handle_chat(user_message, history):
    if history is None:
        history = []
    if not user_message or not user_message.strip():
        yield history
        return

    prior_history = list(history)

    # Show a thinking placeholder while the agent runs
    history = prior_history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": "Thinking..."},
    ]
    yield history

    problem = config_problem()
    if problem:
        history[-1] = {"role": "assistant", "content": f"**Configuration problem:** {problem}"}
        yield history
        return

    # Any failure (bad API key, server crash, network) becomes a chat message
    # instead of a traceback that kills the UI mid-demo.
    try:
        response_text = await chat_with_agent(user_message, prior_history)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        traceback.print_exc()
        response_text = (
            f"**Something went wrong:** `{describe_exception(exc)}`\n\n"
            "Check that your API key and MODEL_NAME in `.env` are valid and that the "
            "MCP server can start (`python src/client.py` tests it without the LLM)."
        )

    history[-1] = {"role": "assistant", "content": response_text}
    yield history


EXAMPLES = [
    "Find me some moody restaurants",
    "Tell me about Iron & Embers",
    "What's a zen dining experience in Little Tokyo?",
]


# Gradio Interface
with gr.Blocks(title="Connoisseur Companion", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Connoisseur Companion\nYour AI guide to California's restaurant scene. Ask me about restaurants by name, cuisine, or vibe!")

    # type="messages" is required — this app yields OpenAI-style {role, content} dicts
    chatbot = gr.Chatbot(height=500, type="messages")
    msg_input = gr.Textbox(
        label="Ask about restaurants",
        placeholder='e.g., "Find me a moody spot in DTLA" or "Tell me about Sakura Garden"',
    )

    with gr.Row():
        example_buttons = [gr.Button(text, size="sm") for text in EXAMPLES]

    # Run the agent, then clear the textbox (clearing in a separate listener would
    # race the handler and could blank the input before it is read).
    msg_input.submit(handle_chat, [msg_input, chatbot], [chatbot]).then(
        lambda: "", None, msg_input
    )

    for button, text in zip(example_buttons, EXAMPLES):
        button.click(lambda t=text: t, None, msg_input).then(
            handle_chat, [msg_input, chatbot], [chatbot]
        ).then(lambda: "", None, msg_input)

# Launch the App
if __name__ == "__main__":
    print("Starting Connoisseur Companion...")

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
    )
