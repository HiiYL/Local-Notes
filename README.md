# Local Notes: Privacy-Focused RAG for Apple Notes

Local Notes is a private, local-first semantic search and Retrieval-Augmented Generation (RAG) tool for your Apple Notes. Notes are retrieved using AppleScript (no cloud). Text is embedded locally with SentenceTransformers and indexed with FAISS via LangChain for fast semantic queries. We use LangChain's SemanticChunker to split text into meaningful chunks.

No data leaves your machine.

## Features
- Apple Notes as a data source via AppleScript (no account credentials needed)
- Local embeddings using `sentence-transformers` (via LangChain `HuggingFaceEmbeddings`)
- Fast vector search with `FAISS` (via LangChain)
- Semantic chunking with LangChain `SemanticChunker`
- Simple CLI using Typer: `index` and `query`
- Incremental indexing with change detection and optional `--since` time filter
- RAG answering over your notes using a local LLM by default (Ollama), swappable to OpenAI
- Extensible data source abstraction for future sources

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

2. Index Apple Notes:

```bash
python -m local_notes.cli index apple-notes --store-dir ./data/index
# Incremental is on by default; use --no-incremental to force full rebuild
# Only re-index notes modified since a given time:
# python -m local_notes.cli index apple-notes --store-dir ./data/index --since "2025-09-01T00:00:00"
```

The first run will download an embedding model (~90MB) and ask macOS to allow Terminal to control Notes. Approve the prompt.

3. Query your notes semantically:

```bash
python -m local_notes.cli query "how to setup postgres" --store-dir ./data/index --k 5
# Show longer snippets or hide text:
# python -m local_notes.cli query "..." --max-chars 600
# python -m local_notes.cli query "..." --no-show-text
```

4. Ask questions with RAG (LLM-generated answer):

By default this uses a local LLM via Ollama (privacy-first). You can switch to OpenAI at any time.

```bash
# Local LLM with Ollama (default model: gemma2)
python -m local_notes.cli ask "summarize my postgres setup steps" --k 6

# Choose a different local model (examples)
python -m local_notes.cli ask "..." --llm-model gemma2:2b
python -m local_notes.cli ask "..." --llm-model llama3.1

# Use OpenAI (requires OPENAI_API_KEY)
export OPENAI_API_KEY=...
python -m local_notes.cli ask "..." --provider openai --llm-model gpt-4o-mini
```

## Data Location
The default index directory is `./data/index`. It contains:
- `index.faiss` – FAISS index
- `index.pkl` – LangChain store metadata

Delete this folder to reset the index.

## Extending Data Sources
Implement `local_notes.datasources.base.DataSource` and register in the CLI. See `local_notes/datasources/apple_notes.py` for a reference implementation.

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

## Configuration

You can set defaults via environment variables to avoid repeating flags:

```bash
# Default local LLM model for `ask`
export OLLAMA_MODEL=gemma2          # or gemma2:2b, llama3.1, etc.

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

## Security & Privacy
- All processing is local.
- No telemetry.
- You control the index files.

## License
MIT
