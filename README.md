# Dine-Discover-AI

Dine-Discover-AI is a conversational ReAct agent designed to act as an expert AI guide to California's vibrant restaurant scene. Built with the Model Context Protocol (MCP), it enables intelligent exploration of structured restaurant details, vibe-based recommendations, and detailed user reviews.

The project is built to be completely **LLM-agnostic**. It uses the universal OpenAI API format, which means you can plug in any open-source or local model (via Ollama, LMStudio, vLLM) or use popular cloud providers (like OpenAI, Groq, TogetherAI, etc.).

## Features

- **Conversational Interface**: Chat directly with the AI via a Gradio web interface.
- **Vibe-Based Recommendations**: Look for restaurants using descriptive keywords like "moody," "romantic," or "zen."
- **Structured Data Retrieval**: Instantly pull details like rating, price range, and signature dishes for specific spots.
- **Detailed Reviews**: Access full user reviews and image descriptions for a deeper understanding of the dining experience.
- **Agentic Architecture**: Powered by Langchain and MCP for modular tool use, allowing the AI to call functions and query data intelligently.

## Architecture

```text
Browser ──► Gradio UI (src/app.py)  ── MCP host + ReAct loop
                 │
                 ├──► LLM  (OpenAI-compatible API)  — decides which tool to call
                 │
                 └──► MCP server (src/server.py, stdio subprocess)
                          ├─ tool: get_restaurant_info   → structured_restaurant_data.json
                          ├─ tool: recommend_by_vibe     → JSON + raw culinary map text
                          ├─ tool: get_review            → augmented_user_review.json (joined on itemId)
                          └─ resource: culinary-map://california
```

## Project Structure

```text
Dine-Discover-AI/
├── .env                         # Environment variables (copy from .env.example)
├── requirements.txt             # Python dependencies
├── src/                         # Source code
│   ├── app.py                   # Main Gradio application & ReAct Agent (MCP host)
│   ├── server.py                # MCP Server exposing tools and resources
│   ├── client.py                # MCP Client demo/utility (no LLM required)
│   └── restaurant_data_management.py # Utility to manage structured JSON data
└── data/                        # Application data
    ├── raw/                     # Original text and image data
    └── processed/               # Extracted & structured JSON databases
```

## Setup & Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/FenilKaneria/Dine-Discover-AI.git
   cd Dine-Discover-AI
   ```

2. **Create a virtual environment** (recommended — keeps Gradio's pins away from your other projects):

   ```bash
   python -m venv .venv
   .venv\Scripts\activate       # Windows
   source .venv/bin/activate    # macOS / Linux
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**

   ```bash
   cp .env.example .env
   ```

   Then open `.env` and set your LLM configuration. See below.

## LLM Configuration

Because this project uses the standard OpenAI SDK and Langchain-OpenAI, you can run it with any model.

**Using OpenAI (Default):**

```env
OPENAI_API_KEY=sk-your-openai-key
MODEL_NAME=gpt-4o-mini
```

**Using Local/Open-Source Models (Ollama, LMStudio, etc):**

```env
OPENAI_API_KEY=ollama            # some local servers require a dummy key
OPENAI_API_BASE=http://localhost:11434/v1
MODEL_NAME=llama3
```

> **Important:** do not wrap values in quotes, and never leave a quote unterminated —
> `python-dotenv` silently skips a malformed line *and the line after it*, which
> makes the key look absent. `python src/app.py` prints a warning at startup if the
> key is missing or too short to be real.

## Usage

**Run the Gradio Application:**

```bash
python src/app.py
```

Open the printed local URL (default <http://127.0.0.1:7860>) to start chatting.

Optional environment overrides:

| Variable      | Default       | Effect                                                     |
|---------------|---------------|------------------------------------------------------------|
| `SHARE`       | `false`       | Set to `true` for a public Gradio tunnel (needs internet). |
| `SERVER_NAME` | `127.0.0.1`   | Set to `0.0.0.0` to expose the UI on your local network.   |
| `MODEL_NAME`  | `gpt-4o-mini` | Model id passed to the OpenAI-compatible endpoint.         |

**Testing the MCP Server/Client (no API key needed):**

```bash
python src/client.py
```

This starts the server over stdio, verifies the 3 tools and 1 resource are
discoverable, and runs one live call against each tool. Use it to prove the data
layer works independently of the LLM.

**Managing the restaurant database:**

```bash
python src/restaurant_data_management.py            # interactive CRUD menu (add uses the LLM)
python src/restaurant_data_management.py --test     # offline unit tests, no network calls
```

## Data Notes

- `structured_restaurant_data.json` — 210 restaurant records with
  `name`, `location`, `type`, `food_style`, `rating`, `price_range`,
  `signatures`, `vibe`, `environment`, `shortcomings`, `itemId`.
- `augmented_user_review.json` — 10 reviews, linked to restaurants by `itemId`
  (not by name). Only the first 10 restaurants currently have reviews;
  `get_review` reports which ones do when a lookup misses.
- `augmented_food_recipe.json` and `data/raw/synthetic_recipe_images/` are
  present in the repo but are not yet used by any MCP tool.
