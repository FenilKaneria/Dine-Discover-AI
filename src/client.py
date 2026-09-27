# Libraries for MCP client, LLM handling, and async operations
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.context import RequestContext
from mcp.types import (
    Root,
    ListRootsResult,
    TextContent,
    CreateMessageResult,
    CreateMessageRequestParams,
    ErrorData,
    INTERNAL_ERROR,
)

# config loads .env and holds the Groq / model settings
sys.path.insert(0, str(Path(__file__).parent))
from config import CHAT_MODEL, groq_client  # noqa: E402

# Configuration
SERVER_SCRIPT = str(Path(__file__).parent / "server.py")
PROJECT_DIR = Path(__file__).parent.parent.resolve()
# StdioServerParameters launches "server.py" via stdin/stdout. sys.executable is
# used instead of "python" so the server inherits this venv/conda environment.
server_params = StdioServerParameters(
    command=sys.executable,
    args=[SERVER_SCRIPT],
    env=os.environ.copy(),
)


# ROOTS — Tell the server which directories it can access
async def handle_list_roots(context: RequestContext) -> ListRootsResult:
    """Limit the server's file access to this project directory."""
    # .as_uri() produces a properly escaped file:// URL on Windows and POSIX alike.
    return ListRootsResult(roots=[Root(uri=PROJECT_DIR.as_uri(), name=PROJECT_DIR.name)])


def list_roots() -> list[Root]:
    """Synchronous helper used for printing the configured roots."""
    return [Root(uri=PROJECT_DIR.as_uri(), name=PROJECT_DIR.name)]


# SAMPLING — Handle LLM requests delegated from the server
async def handle_sampling(
    context: RequestContext, params: CreateMessageRequestParams
) -> CreateMessageResult | ErrorData:
    """Run an LLM call on behalf of the server and return the result."""
    try:
        # The OpenAI client is built lazily so this script still runs the
        # tool demos when no API key is configured.
        openai_client = groq_client()

        # Extract the prompt text from the first sampling message
        content = params.messages[0].content
        prompt = getattr(content, "text", None) or str(content)

        print("\n[Sampling] Server requested LLM task:")
        print(f"  Prompt preview: {prompt[:150]}...")

        model_name = CHAT_MODEL
        response = openai_client.chat.completions.create(
            model=model_name,
            max_tokens=params.maxTokens or 200,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = response.choices[0].message.content or ""
        print(f"  LLM Response: {response_text[:100]}...")

        return CreateMessageResult(
            role="assistant",
            content=TextContent(type="text", text=response_text),
            model=model_name,
        )
    except Exception as exc:
        # Returning an ErrorData keeps the protocol conversation valid; raising
        # here would tear down the whole session.
        print(f"  [Sampling] failed: {type(exc).__name__}: {exc}")
        return ErrorData(code=INTERNAL_ERROR, message=f"Sampling failed: {exc}")


# HELPER — Call a tool on an open session and return the parsed JSON result
async def call_tool(session: ClientSession, tool_name: str, arguments: dict) -> Any:
    """Call a tool and return the parsed JSON result (or the raw text if not JSON)."""
    result = await session.call_tool(tool_name, arguments=arguments)

    if getattr(result, "isError", False):
        return {"status": "error", "message": f"Server reported an error for {tool_name}."}
    if not result.content:
        return {"status": "error", "message": f"{tool_name} returned no content."}

    text = getattr(result.content[0], "text", None)
    if text is None:
        return {"status": "error", "message": f"{tool_name} returned non-text content."}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"status": "raw", "text": text}


# CONNECTION & DISCOVERY
async def verify_connection(session: ClientSession) -> None:
    """Verify all expected tools and resources exist on the server."""
    print("=" * 60)
    print("MCP Connection Verification")
    print("=" * 60)

    # list_tools() sends a "tools/list" JSON-RPC request to the server
    tools_result = await session.list_tools()
    tool_names = [tool.name for tool in tools_result.tools]
    print(f"\nDiscovered {len(tool_names)} tools:")
    for tool in tools_result.tools:
        description = (tool.description or "").replace("\n", " ")
        print(f"  - {tool.name}: {description[:80]}...")

    for required in (
        "get_restaurant_info",
        "recommend_by_vibe",
        "get_review",
        "search_knowledge_base",
        "search_images",
    ):
        assert required in tool_names, f"FAIL: {required} not found!"
    print("\nAll required tools verified!")

    # list_resources() discovers data endpoints the server exposes
    resources_result = await session.list_resources()
    print(f"\nDiscovered {len(resources_result.resources)} resources:")
    for resource in resources_result.resources:
        print(f"  - {resource.uri}: {resource.name}")

    roots = list_roots()
    print(f"\nConfigured {len(roots)} roots:")
    for root in roots:
        print(f"  - {root.name}: {root.uri}")



# DEMOS — Call each tool through the MCP protocol
async def demo_get_restaurant_info(session: ClientSession) -> None:
    """Demo: Look up a restaurant by name."""
    print("\n" + "-" * 60)
    print("Demo: get_restaurant_info('Iron & Embers')")
    print("-" * 60)

    data = await call_tool(session, "get_restaurant_info", {"restaurant_name": "Iron & Embers"})
    print(json.dumps(data, indent=2))


async def demo_recommend_by_vibe(session: ClientSession) -> None:
    """Demo: Hybrid RAG recommendation with a filter."""
    print("\n" + "-" * 60)
    print("Demo: recommend_by_vibe('moody date night', max_price=3)")
    print("-" * 60)

    data = await call_tool(session, "recommend_by_vibe", {"vibe": "moody date night", "max_price": 3})
    print(f"Query: {data.get('query')}  |  Retrieval: {data.get('retrieval')}")
    for match in data.get("results", []):
        print(
            f"  - {match['name']} ({match['cuisine']}) - {match['rating']}/5, "
            f"{match['price_range']} in {match['location']}"
        )


async def demo_search_knowledge_base(session: ClientSession) -> None:
    """Demo: Semantic search over culinary-map prose and reviews."""
    print("\n" + "-" * 60)
    print("Demo: search_knowledge_base('dining room full of plants and greenery')")
    print("-" * 60)

    data = await call_tool(session, "search_knowledge_base", {"query": "dining room full of plants and greenery"})
    for passage in data.get("passages", []):
        print(f"  - [{passage['source']}] {passage['restaurant']} (sim {passage['similarity']})")


async def demo_search_images(session: ClientSession) -> None:
    """Demo: Text-to-image search with jina-clip-v2."""
    print("\n" + "-" * 60)
    print("Demo: search_images('bowl of ramen with a soft-boiled egg')")
    print("-" * 60)

    data = await call_tool(session, "search_images", {"query": "bowl of ramen with a soft-boiled egg"})
    for image in data.get("images", []):
        print(f"  - {image['name']} (sim {image['similarity']}): {image['uri']}")


async def demo_get_review(session: ClientSession) -> None:
    """Demo: Retrieve a restaurant review."""
    print("\n" + "-" * 60)
    print("Demo: get_review('Iron & Embers')")
    print("-" * 60)

    data = await call_tool(session, "get_review", {"restaurant_name": "Iron & Embers"})
    print(json.dumps(data, indent=2))


# Main Entry Point
async def main() -> int:
    """Open a single session and run every demo through it."""
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(
                read,
                write,
                sampling_callback=handle_sampling,
                list_roots_callback=handle_list_roots,
            ) as session:
                await session.initialize()

                await verify_connection(session)
                await demo_get_restaurant_info(session)
                await demo_recommend_by_vibe(session)
                await demo_get_review(session)
                await demo_search_knowledge_base(session)
                await demo_search_images(session)

        print("\nAll MCP demos completed successfully.")
        return 0
    except AssertionError as exc:
        print(f"\nVerification failed: {exc}")
        return 1
    except Exception as exc:
        # anyio wraps errors from inside the session in an ExceptionGroup.
        while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
            exc = exc.exceptions[0]
        print(f"\nCould not run the MCP demos: {type(exc).__name__}: {exc}")
        print("Check that ./data exists and that `python src/server.py` starts without errors.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
