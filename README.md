# Local Notes: Privacy-Focused RAG for Apple Notes

Local Notes is a private, local-first semantic search and Retrieval-Augmented Generation (RAG) tool for your Apple Notes. Notes are retrieved using AppleScript (no cloud). Text is embedded locally with SentenceTransformers and indexed with FAISS via LangChain for fast semantic queries. We use LangChain's SemanticChunker to split text into meaningful chunks.

No data leaves your machine.

## Features
- **Apple Notes source** via AppleScript (no account credentials needed)
- **Local embeddings** with `sentence-transformers` (LangChain `HuggingFaceEmbeddings`)
- **FAISS semantic search** (LangChain)
- **Semantic chunking** with LangChain `SemanticChunker`
- **Hybrid retrieval**: vector + lexical (RapidFuzz/Jaccard) with Reciprocal Rank Fusion
- **MMR diversity** applied earlier and hybrid fusion for improved coverage
- **Streaming RAG** answers with live citations ([n]) and Markdown rendering
- **Web UI** (ChatGPT-like) with conversation history, source chips, inline snippet expand + copy
- **Conversations** persisted in SQLite (`conversations.db`) with citations stored per message
- **Stable IDs** in metadata and citations: `doc_id`, `chunk_id` so citations survive re-indexing or title changes
- **Embedding cache** (SQLite) avoids re-embedding unchanged text
- **Incremental indexing** with change detection and optional `--since` filter
- **Simple CLI**: `index`, `query`, and `ask` (streaming by default)
- **Privacy-first**: default local LLM via Ollama; can switch to OpenAI
- **Extensible datasources**

## Requirements
- macOS with Apple Notes app
- Python 3.10+
- `uv` for dependency management (install: `curl -LsSf https://astral.sh/uv/install.sh | sh` and restart shell)
- Permissions to allow Terminal/iTerm to automate Notes (System Settings > Privacy & Security > Automation / Accessibility)
- (Optional, for local LLM) Ollama installed and running for best privacy

## Quick Start

1. Create and activate a virtual environment with `uv` and install dependencies:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Architecture

At a high level, Local Notes consists of a data ingestion/indexing pipeline, a retrieval/generation service layer, a FastAPI server that exposes both API and a Web UI, and a small SQLite store for conversations and citations.

### Components

- **Data source (`local_notes/datasources/`)**
  - Reads Apple Notes via AppleScript.
  - Produces `Document` objects with `id`, `title`, `source`, `metadata`, and full `text`.

- **Indexing (`local_notes/indexing/pipeline.py`)**
  - Splits text with LangChain `SemanticChunker`.
  - Embeds with `HuggingFaceEmbeddings` wrapped by `CachedEmbeddings` (SQLite cache) to avoid recomputation.
  - Stores vectors in FAISS and metadata per chunk:
    - Stable IDs: `doc_id`, `chunk_id`
    - Also `title`, `source`, `chunk`, `doc_hash`, plus any datasource metadata
  - Incremental by default: only changed docs (by `doc_hash`) are re-indexed.

- **Service layer (`local_notes/service.py`)**
  - Hybrid retrieval (vector + lexical) with Reciprocal Rank Fusion.
  - MMR/Hybrid diversity for better coverage in `search_index()` and RAG.
  - Builds prompts and streams tokens from your chosen provider (Ollama/OpenAI).
  - Emits SSE events: `sources` (all), `citations` (cited-only), `delta`, `done`.

- **Server (`local_notes/server.py`)**
  - FastAPI endpoints: `/search`, `/ask/stream`, and conversation CRUD + `/conv/{id}/ask/stream`.
  - Serves Web UI static assets at `/` and `/static`.

- **Web UI (`local_notes/web/`)**
  - Chat-like interface with streaming answers, live citations, snippet chips (expand/copy), settings.
  - Persists settings in localStorage.

- **Conversations store (`local_notes/storage/conversations.py`)**
  - SQLite DB with tables `conversations` and `messages`.
  - Saves assistant messages and citations (including `doc_id`, `chunk_id`).

### Data Flow

```mermaid
flowchart TD
  A[Apple Notes] -- AppleScript --> B[DataSource]
  B --> C[Documents (id, title, text, metadata)]
  C --> D[SemanticChunker]
  D --> E[CachedEmbeddings -> HF Embeddings]
  E --> F[FAISS Index + Metadata (doc_id, chunk_id, ...)]

  subgraph Server
    G[Service: Hybrid Retrieval + Prompt Builder]
    H[LLM Provider: Ollama/OpenAI]
    I[Citers: sources/citations events]
  end

  F --> G
  G --> H
  H --> G
  G --> I

  I --> J[FastAPI SSE / Web UI]
  J --> K[Web Chat: messages, chips, expand/copy]
  K --> L[SQLite Conversations (messages + citations)]
```

2. Index Apple Notes (with embedding cache & incremental updates):

```bash
python -m local_notes.cli index apple-notes --store-dir ./data/index
# Incremental is on by default; use --no-incremental to force full rebuild
# Only re-index notes modified since a given time:
# python -m local_notes.cli index apple-notes --store-dir ./data/index --since "2025-09-01T00:00:00"
```

The first run will download an embedding model (~90MB) and ask macOS to allow Terminal to control Notes. Approve the prompt.

3. Query your notes semantically (hybrid retrieval):

```bash
python -m local_notes.cli query "how to setup postgres" --store-dir ./data/index --k 5
# Show longer snippets or hide text:
# python -m local_notes.cli query "..." --max-chars 600
# python -m local_notes.cli query "..." --no-show-text
```

4. Ask questions with RAG (LLM-generated, streaming by default):

By default this uses a local LLM via Ollama (privacy-first). You can switch to OpenAI at any time.

```bash
# Local LLM with Ollama (default model: gemma2), streams tokens by default
python -m local_notes.cli ask "summarize my postgres setup steps" --k 6

# Choose a different local model (examples)
python -m local_notes.cli ask "..." --llm-model gemma2:2b
python -m local_notes.cli ask "..." --llm-model llama3.1

# Use OpenAI (requires OPENAI_API_KEY)
export OPENAI_API_KEY=...
python -m local_notes.cli ask "..." --provider openai --llm-model gpt-4o-mini
```

## Data Location
Default index directory: `./data/index/`

Files created:
- `index.faiss` – FAISS index
- `index.pkl` – LangChain store metadata
- `emb_cache.sqlite` – SQLite embedding cache keyed by (model, md5(content))

Conversations database (web UI):
- `./data/conversations.db`

Delete these to reset state.

## Extending Data Sources
Implement `local_notes.datasources.base.DataSource` and register in the CLI. See `local_notes/datasources/apple_notes.py` for a reference implementation.

## Web UI

Start the API server and open the chat UI:

```bash
uvicorn local_notes.server:app --reload --port 8000
# then open http://127.0.0.1:8000/
```

Web UI highlights:
- **Streaming** assistant messages with **live citations**; chips show snippet on hover.
- **Click chips** to expand full snippet inline; **Copy** snippet.
- **Settings** gear toggles controls (Provider, Model, Top K).
- **Conversations** auto-saved; citations persisted with stable IDs.

## Local LLM with Ollama

Ollama is the default LLM runtime for the `ask` command.

```bash
# Install (macOS)
brew install ollama

# Start service
ollama serve

# Pull a recommended model (9B default)
ollama pull gemma2   # default in this app
# or a faster, smaller variant
ollama pull gemma2:2b

# Ask (uses the default model unless overridden by --llm-model)
python -m local_notes.cli ask "..."
```

You can override the default with `--llm-model` or set an env var:

```bash
export OLLAMA_MODEL=gemma2

## API Endpoints

Served by FastAPI when you run `uvicorn`:

- `GET /` – Web UI (served static)
- `GET /search` – Search with hybrid retrieval
  - Query params: `q`, `k`, `max_chars`, `store_dir`, `embed_model`
- `POST /ask/stream` – SSE streaming for answers (no conversation)
- `GET /conv` – List conversations
- `POST /conv` – Create conversation `{ id, title }`
- `GET /conv/{id}/messages` – Get messages (oldest→newest)
- `POST /conv/{id}/ask/stream` – SSE streaming tied to a conversation; persists user + assistant with citations
- `GET /conv/{id}/export` – Export conversation + messages
- `POST /conv/import` – Import conversation
- `DELETE /conv/{id}` – Delete conversation

## Configuration

You can set defaults via environment variables to avoid repeating flags:

```bash
# Default local LLM model for `ask` (Ollama)
export OLLAMA_MODEL=gemma2          # or gemma2:2b, gemma3:12b-it-qat, llama3.1, etc.

# Use OpenAI instead (for `ask` with --provider openai)
export OPENAI_API_KEY=sk-...

# Optional: choose a different index directory by default (used in examples)
# Not strictly required since the CLI always accepts --store-dir
```
```

## OpenAI (optional)

To use OpenAI for the `ask` command:

```bash
export OPENAI_API_KEY=... 
python -m local_notes.cli ask "..." --provider openai --llm-model gpt-4o-mini
```

## Troubleshooting
- **AppleScript returns 0 notes**
  - Open the Notes app once manually and ensure you have at least one account with notes.
  - Check Automation permissions: System Settings > Privacy & Security > Automation. Allow your Terminal/IDE to control Notes.

- **Ollama: server not responding**
  - Start the server: `brew services start ollama` or run `ollama serve` in a terminal.
  - Verify: `curl http://localhost:11434/api/tags` should return JSON.

- **Ollama: pull model manifest: file does not exist**
  - Use valid tags. For Gemma 2, common tags are `gemma2` (9B default) and `gemma2:2b`.
  - Example: `ollama pull gemma2` or `ollama pull gemma2:2b`.

- **OpenAI setup**
  - Export `OPENAI_API_KEY` and use `--provider openai`.

- **Chunk boundaries look odd in snippets**
  - Increase retrieval depth `--k` or display length `--max-chars` in `query`.

- **Streaming text looks oddly formatted**
  - The UI renders Markdown with GFM and normalization, but some model outputs collapse bullets without newlines. Try sending again or adjust the prompt; we normalize common cases.

- **Conversation messages blank after streaming**
  - We now persist the assistant message before emitting `done` and finalize bubbles client-side without refresh. Hard-refresh the UI to load the latest scripts.

## Indexing Details

- **Stable IDs in metadata**
  - `doc_id` and `chunk_id` added per chunk. Citations store these so they are stable across re-indexes.

- **Embedding Cache**
  - SQLite cache (`emb_cache.sqlite`) avoids recomputation. Keys: `(model, md5(text))`.

- **Incremental Indexing**
  - We md5-hash full note content (`doc_hash`) and only re-embed changed notes; unchanged chunks are retained. `--since` limits to recently modified notes.

- **Hybrid Retrieval**
  - Vector similarities fused with a lexical ranker (RapidFuzz/Jaccard) using Reciprocal Rank Fusion, improving exact-term recall while keeping semantic relevance.

## Security & Privacy
- All processing is local.
- No telemetry.
- You control the index files.

## License
MIT
