import json
import os
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .datasources.apple_notes import AppleNotesDataSource, AppleNotesIncremental
from .indexing.pipeline import build_index
from .utils.html import html_to_text
from .utils.dates import parse_to_unix_ts
from .models import Document
from .llm.providers import get_llm
from .service import search_index, ask_question
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

app = typer.Typer(help="Local Notes: private semantic search for Apple Notes")
console = Console()


def ensure_store_dir(path: str):
    os.makedirs(path, exist_ok=True)


@app.command()
def index(
    source: str = typer.Argument("apple-notes", help="Data source to index (currently only 'apple-notes')"),
    store_dir: str = typer.Option("./data/index", help="Directory to store the vector index"),
    model_name: str = typer.Option("sentence-transformers/all-MiniLM-L6-v2", help="SentenceTransformers model"),
    incremental: bool = typer.Option(True, help="Incremental indexing: only notes changed since last index are updated"),
    since: str = typer.Option("", help="Only consider notes modified since this time (ISO or natural language; optional)"),
):
    """Index data from the selected source."""
    ensure_store_dir(store_dir)
    if source != "apple-notes":
        raise typer.BadParameter("Only 'apple-notes' is implemented currently")

    # Two-pass approach: list metadata first, then fetch bodies only for changed/new notes
    console.print("Listing Apple Notes metadata...", style="bold cyan")
    inc = AppleNotesIncremental()
    meta = inc.list_metadata()

    # Optional since filter (post-list) to reduce processing
    since_ts = parse_to_unix_ts(since) if since else None
    if since_ts is not None:
        def m_ts(m):
            return parse_to_unix_ts(m.get("modified", "")) or 0
        meta = [m for m in meta if m_ts(m) >= since_ts]

    # Determine which notes changed vs. what's in FAISS
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    index_files = ["index.faiss", "index.pkl"]
    index_exists = all(os.path.exists(os.path.join(store_dir, f)) for f in index_files)
    existing_modified: dict[str, str] = {}
    existing_doc_ids: set[str] = set()
    if index_exists:
        vs = FAISS.load_local(store_dir, embeddings, allow_dangerous_deserialization=True)
        # read one metadata entry per doc_id (any chunk) to get last known modified
        for _id, doc in vs.docstore._dict.items():  # type: ignore[attr-defined]
            md = getattr(doc, "metadata", {}) or {}
            did = md.get("doc_id")
            if not did:
                continue
            existing_doc_ids.add(did)
            if did not in existing_modified and "modified" in md:
                existing_modified[did] = md["modified"]

        # Deletions: if a note no longer exists in Apple Notes, remove it from FAISS
        current_ids = {m["id"] for m in meta}
        missing = existing_doc_ids - current_ids
        if missing:
            ids_to_delete = []
            for k, d in vs.docstore._dict.items():  # type: ignore[attr-defined]
                md = getattr(d, "metadata", {}) or {}
                if md.get("doc_id") in missing:
                    ids_to_delete.append(k)
            if ids_to_delete:
                vs.delete(ids=ids_to_delete)
                vs.save_local(store_dir)

    changed_ids = []
    for m in meta:
        did = m["id"]
        prev_mod = existing_modified.get(did)
        if prev_mod is None:
            changed_ids.append(did)
        elif m.get("modified") != prev_mod:
            changed_ids.append(did)

    console.print(f"Notes needing body fetch: {len(changed_ids)}", style="bold cyan")

    # Fetch bodies only for changed/new ids
    bodies = inc.fetch_bodies(changed_ids)
    body_by_id = {b["id"]: b["body_html"] for b in bodies}

    # Build Document list: for changed/new ones we use fresh body; for others, skip (incremental will keep them)
    docs = []
    for m in meta:
        did = m["id"]
        if did not in body_by_id:
            continue
        body_md = html_to_text(body_by_id[did])
        title = m.get("title") or (body_md.split("\n", 1)[0] if body_md else "Untitled")
        docs.append(
            Document(
                id=did,
                title=title,
                text=body_md,
                source="apple-notes",
                metadata={"folder": m.get("folder", ""), "modified": m.get("modified", "")},
            )
        )

    console.print(f"Upserting {len(docs)} notes into index...", style="bold cyan")
    build_index(docs, store_dir=store_dir, model_name=model_name, incremental=incremental)
    console.print("Index build complete.", style="bold green")


@app.command()
def query(
    text: str = typer.Argument(..., help="Query text"),
    store_dir: str = typer.Option("./data/index", help="Directory where the vector index is stored"),
    model_name: str = typer.Option("sentence-transformers/all-MiniLM-L6-v2", help="SentenceTransformers model"),
    k: int = typer.Option(5, help="Top K results"),
    show_text: bool = typer.Option(True, help="Display text snippets for each result"),
    max_chars: int = typer.Option(300, help="Maximum characters to show per result snippet"),
):
    """Query the semantic index."""
    index_files = ["index.faiss", "index.pkl"]
    if not all(os.path.exists(os.path.join(store_dir, f)) for f in index_files):
        raise typer.BadParameter(f"No FAISS index found in {store_dir}. Run 'index' first.")

    results = search_index(text, store_dir=store_dir, embed_model=model_name, k=k, max_chars=max_chars)

    table = Table(title="Search Results")
    table.add_column("Rank", justify="right")
    table.add_column("Title")
    table.add_column("Folder")
    table.add_column("Chunk")
    table.add_column("Score")
    if show_text:
        table.add_column("Text")

    for r in results:
        if show_text:
            table.add_row(str(r["rank"]), r["title"], str(r["folder"]), str(r["chunk"]), f"{r['score']:.4f}", r["text"])
        else:
            table.add_row(str(r["rank"]), r["title"], str(r["folder"]), str(r["chunk"]), f"{r['score']:.4f}")

    console.print(table)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Your question to ask over your notes"),
    store_dir: str = typer.Option("./data/index", help="Directory where the vector index is stored"),
    embed_model: str = typer.Option("sentence-transformers/all-MiniLM-L6-v2", help="Embedding model used to build the index"),
    k: int = typer.Option(5, help="Top K chunks to retrieve"),
    provider: str = typer.Option("ollama", help="LLM provider: ollama or openai"),
    llm_model: str = typer.Option("", help="LLM model name (defaults: ollama=gemma2, openai=gpt-4o-mini)"),
    stream: bool = typer.Option(True, help="Stream tokens as they are generated"),
):
    """Ask a question using RAG over your indexed Apple Notes.

    Retrieval uses the FAISS index; generation uses a local (Ollama) model by default, and can be swapped to OpenAI.
    """
    index_files = ["index.faiss", "index.pkl"]
    if not all(os.path.exists(os.path.join(store_dir, f)) for f in index_files):
        raise typer.BadParameter(f"No FAISS index found in {store_dir}. Run 'index' first.")

    if stream:
        from .service import stream_answer
        console.rule("Answer")
        full = []
        srcs = None
        for ev, data in stream_answer(
            question=question,
            store_dir=store_dir,
            embed_model=embed_model,
            k=k,
            provider=provider,
            llm_model=(llm_model or None),
        ):
            if ev == "sources":
                srcs = json.loads(data)
                # Print sources first
                console.rule("Sources")
                src_table = Table(show_header=True, header_style="bold")
                src_table.add_column("#", justify="right")
                src_table.add_column("Title")
                src_table.add_column("Folder")
                src_table.add_column("Chunk")
                for s in srcs:
                    src_table.add_row(str(s["rank"]), s["title"], str(s["folder"]), str(s["chunk"]))
                console.print(src_table)
                console.rule("Answer (stream)")
            elif ev == "delta":
                print(data, end="", flush=True)
                full.append(data)
        print()
    else:
        answer_text, sources = ask_question(
            question=question,
            store_dir=store_dir,
            embed_model=embed_model,
            k=k,
            provider=provider,
            llm_model=(llm_model or None),
        )

        console.rule("Answer")
        console.print(answer_text)
        console.rule("Sources")
        src_table = Table(show_header=True, header_style="bold")
        src_table.add_column("#", justify="right")
        src_table.add_column("Title")
        src_table.add_column("Folder")
        src_table.add_column("Chunk")
        for s in sources:
            src_table.add_row(str(s["rank"]), s["title"], str(s["folder"]), str(s["chunk"]))
        console.print(src_table)

if __name__ == "__main__":
    app()
