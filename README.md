# Dine-Discover-AI

Dine-Discover-AI is a conversational ReAct agent designed to act as an expert AI guide to California's vibrant restaurant scene. Built with the Model Context Protocol (MCP), it enables intelligent exploration of structured restaurant details, vibe-based recommendations, and detailed user reviews.

The project is built to be completely **LLM-agnostic**. It uses the universal OpenAI API format, which means you can plug in any open-source or local model (via Ollama, LMStudio, vLLM) or use popular cloud providers (like OpenAI, Groq, TogetherAI, etc.).

## Features

- **Conversational Interface**: Chat directly with the AI via a beautiful Gradio web interface.
- **Vibe-Based Recommendations**: Look for restaurants using descriptive keywords like "moody," "romantic," or "zen" (semantic search).
- **Structured Data Retrieval**: Instantly pull details like rating, price range, and signature dishes for specific spots.
- **Detailed Reviews**: Access full user reviews and image descriptions for a deeper understanding of the dining experience.
- **Agentic Architecture**: Powered by Langchain and MCP for modular tool use, allowing the AI to call functions and query data intelligently.

## Project Structure

```
Dine-Discover-AI/
├── .env                         # Environment variables (copy from .env.example)
├── requirements.txt             # Python dependencies
├── src/                         # Source code
│   ├── app.py                   # Main Gradio application & ReAct Agent
│   ├── server.py                # MCP Server exposing tools and resources
│   ├── client.py                # MCP Client demo/utility
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

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   Create an `.env` file from the example:
   ```bash
   cp .env.example .env
   ```
   Open the `.env` file and set up your LLM configuration. See the section below for details.

## LLM Configuration

Because this project uses the standard OpenAI SDK and Langchain-OpenAI, you can run it with any model!

**Using OpenAI (Default):**
Set the following in your `.env`:
```env
OPENAI_API_KEY=sk-your-openai-key
MODEL_NAME=gpt-4o-mini
```

**Using Local/Open-Source Models (Ollama, LMStudio, etc):**
If you have a local server running (for instance, Ollama on port 11434), you can point the app to it by changing the base URL.
```env
OPENAI_API_KEY=ollama  # some local servers require a dummy key
OPENAI_API_BASE=http://localhost:11434/v1
MODEL_NAME=llama3
```

## Usage

**Run the Gradio Application:**
```bash
python src/app.py
```
This will start the Dine-Discover-AI interface. Open the provided local URL in your browser to start chatting!

**Testing the MCP Server/Client:**
If you want to test the server tools individually without the Gradio UI:
```bash
python src/client.py
```
