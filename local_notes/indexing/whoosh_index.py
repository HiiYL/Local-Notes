import os
from typing import Iterable, List, Dict, Any

from whoosh import index
from whoosh.fields import Schema, ID, TEXT, NUMERIC
from whoosh.analysis import StemmingAnalyzer
from whoosh.qparser import MultifieldParser
from whoosh import scoring


SCHEMA = Schema(
    doc_key=ID(stored=True, unique=True),
    title=TEXT(stored=True, analyzer=StemmingAnalyzer()),
    folder=TEXT(stored=True, analyzer=StemmingAnalyzer()),
    heading=TEXT(stored=True, analyzer=StemmingAnalyzer()),
    updated_at=NUMERIC(stored=True, sortable=True),
    content=TEXT(stored=False, analyzer=StemmingAnalyzer()),
)


def _ix_path(store_dir: str) -> str:
    return os.path.join(store_dir, "whoosh")


def open_or_create(store_dir: str):
    path = _ix_path(store_dir)
    os.makedirs(path, exist_ok=True)
    if not index.exists_in(path):
        return index.create_in(path, SCHEMA)
    return index.open_dir(path)


def delete_docs(store_dir: str, doc_keys: Iterable[str]) -> None:
    ix = open_or_create(store_dir)
    with ix.writer() as w:
        for k in doc_keys:
            w.delete_by_term("doc_key", k)


def add_docs(store_dir: str, rows: Iterable[Dict[str, Any]]) -> None:
    """rows: dicts with keys doc_key, title, folder, heading, updated_at, content"""
    ix = open_or_create(store_dir)
    with ix.writer(limitmb=256, procs=0) as w:
        for r in rows:
            w.update_document(
                doc_key=str(r.get("doc_key", "")),
                title=str(r.get("title", "")),
                folder=str(r.get("folder", "")),
                heading=str(r.get("heading", "")),
                updated_at=int(r.get("updated_at", 0) or 0),
                content=str(r.get("content", "")),
            )


def search(store_dir: str, query: str, k: int) -> List[Dict[str, Any]]:
    ix = open_or_create(store_dir)
    bm25f = scoring.BM25F(B=0.75, K1=1.5)
    fieldboosts = {"title": 3.0, "heading": 2.0, "content": 1.0, "folder": 0.5}
    with ix.searcher(weighting=bm25f) as s:
        parser = MultifieldParser(["title", "heading", "folder", "content"], schema=ix.schema, fieldboosts=fieldboosts)
        q = parser.parse(query)
        results = s.search(q, limit=k)
        out: List[Dict[str, Any]] = []
        for r in results:
            out.append({
                "doc_key": r.get("doc_key"),
                "title": r.get("title"),
                "folder": r.get("folder"),
                "heading": r.get("heading"),
                "updated_at": int(r.get("updated_at") or 0),
                "score": float(getattr(r, 'score', 0.0)),
            })
        return out
