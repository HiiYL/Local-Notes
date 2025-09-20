import json
import os
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .datasources.apple_notes import AppleNotesDataSource, AppleNotesIncremental
from .indexing.pipeline import build_index
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from .utils.html import html_to_text
from .utils.dates import parse_to_unix_ts
from .models import Document

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
):
    """Query the semantic index."""
    index_files = ["index.faiss", "index.pkl"]
    if not all(os.path.exists(os.path.join(store_dir, f)) for f in index_files):
        raise typer.BadParameter(f"No FAISS index found in {store_dir}. Run 'index' first.")

    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vs = FAISS.load_local(store_dir, embeddings, allow_dangerous_deserialization=True)

    docs_and_scores = vs.similarity_search_with_score(text, k=k)

    table = Table(title="Search Results")
    table.add_column("Rank", justify="right")
    table.add_column("Title")
    table.add_column("Folder")
    table.add_column("Chunk")
    table.add_column("Score")

    for rank, (doc, score) in enumerate(docs_and_scores, start=1):
        m = doc.metadata or {}
        table.add_row(str(rank), m.get("title", ""), str(m.get("folder", "")), str(m.get("chunk", "")), f"{score:.4f}")

    console.print(table)


if __name__ == "__main__":
    app()
