# Local Notes: Privacy-Focused RAG for Apple Notes

Local Notes is a private, local-first semantic search and Retrieval-Augmented Generation (RAG) pipeline for your Apple Notes. Notes are retrieved using AppleScript (no cloud) and embedded locally with HuggingFace SentenceTransformers, then indexed with FAISS via LangChain for fast semantic queries. Text is split using LangChain's SemanticChunker for higher-quality chunks.

No data leaves your machine.

## Features
- Apple Notes as a data source via AppleScript (no account credentials needed)
- Local embeddings using `sentence-transformers` (via LangChain `HuggingFaceEmbeddings`)
- Fast vector search with `FAISS` (via LangChain)
- Semantic chunking with LangChain `SemanticChunker`
- Simple CLI using Typer: `index` and `query`
- Extensible data source abstraction for future sources

## Requirements
- macOS with Apple Notes app
- Python 3.10+
- `uv` for dependency management (install: `curl -LsSf https://astral.sh/uv/install.sh | sh` and restart shell)
- Permissions to allow Terminal/iTerm to automate Notes (System Settings > Privacy & Security > Automation / Accessibility)

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
```

The first run will download an embedding model (~90MB) and ask macOS to allow Terminal to control Notes. Approve the prompt.

3. Query your notes semantically:

```bash
python -m local_notes.cli query "how to setup postgres" --store-dir ./data/index --k 5
```

## Data Location
The default index directory is `./data/index`. It contains:
- `index.faiss` – FAISS index
- `index.pkl` – LangChain store metadata

Delete this folder to reset the index.

## Extending Data Sources
Implement `local_notes.datasources.base.DataSource` and register in the CLI. See `local_notes/datasources/apple_notes.py` for a reference implementation.

## Troubleshooting
- If AppleScript errors or returns 0 notes, open the Notes app once manually and ensure you have at least one account with notes.
- If Automation permissions were denied, go to System Settings > Privacy & Security > Automation and enable Terminal (or your IDE) for Notes.
- AppleScript output size: very large notes may be truncated by AppleScript; this implementation chunks text after extraction using SemanticChunker, not during AppleScript.

## Security & Privacy
- All processing is local.
- No telemetry.
- You control the index files.

## License
MIT
