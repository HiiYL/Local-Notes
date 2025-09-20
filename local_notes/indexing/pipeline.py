from typing import List

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document as LCDocument

from ..models import Document
from .cache import CachedEmbeddings
import os
import hashlib


def build_index(
    documents: List[Document],
    store_dir: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    incremental: bool = True,
):
    """Build a FAISS index using LangChain with semantic chunking.

    - Converts input `Document` objects to LangChain `Document`s.
    - Uses `SemanticChunker` to split text semantically.
    - Embeds with `HuggingFaceEmbeddings` and persists FAISS to `store_dir`.
    """
    # Wrap embeddings with a SQLite-backed cache to avoid recomputation
    base_embeddings = HuggingFaceEmbeddings(model_name=model_name)
    embeddings = CachedEmbeddings(base_embeddings, cache_path=os.path.join(store_dir, "emb_cache.sqlite"))
    splitter = SemanticChunker(embeddings)

    index_files = ["index.faiss", "index.pkl"]
    index_exists = all(os.path.exists(os.path.join(store_dir, f)) for f in index_files)

    vs = None
    if index_exists:
        vs = FAISS.load_local(store_dir, embeddings, allow_dangerous_deserialization=True)

    # Build mapping of doc_id -> existing doc_hash (if any) and ids to delete
    existing_doc_hash: dict[str, str] = {}
    existing_ids_by_doc: dict[str, list[str]] = {}
    if vs is not None:
        for doc_id, doc in vs.docstore._dict.items():  # type: ignore[attr-defined]
            meta = getattr(doc, "metadata", {}) or {}
            did = meta.get("doc_id")
            if not did:
                continue
            existing_ids_by_doc.setdefault(did, []).append(doc_id)
            dh = meta.get("doc_hash")
            if dh and did not in existing_doc_hash:
                existing_doc_hash[did] = dh

    # Prepare upserts
    upsert_docs: list[LCDocument] = []
    upsert_ids: list[str] = []

    for d in documents:
        if not d.text:
            continue
        # Compute a per-note hash of the full content
        doc_hash = hashlib.md5(d.text.encode("utf-8")).hexdigest()

        if incremental and index_exists:
            if existing_doc_hash.get(d.id) == doc_hash:
                # No change, skip this note
                continue
            # If changed, delete existing chunks for this note
            ids_to_delete = existing_ids_by_doc.get(d.id, [])
            if ids_to_delete:
                vs.delete(ids=ids_to_delete)

        chunks = splitter.split_text(d.text)
        for i, ch in enumerate(chunks):
            meta = {
                "doc_id": d.id,
                "title": d.title,
                "source": d.source,
                "chunk": i,
                "chunk_id": i,
                "doc_hash": doc_hash,
                **d.metadata,
            }
            upsert_docs.append(LCDocument(page_content=ch, metadata=meta))
            upsert_ids.append(f"{d.id}::{i}")

    if not upsert_docs:
        # Nothing to add/modify
        if vs is not None:
            vs.save_local(store_dir)
        else:
            os.makedirs(store_dir, exist_ok=True)
        return vs

    if vs is None:
        # Fresh index
        vs = FAISS.from_documents(upsert_docs, embedding=embeddings, ids=upsert_ids)
    else:
        vs.add_documents(upsert_docs, ids=upsert_ids)

    vs.save_local(store_dir)
    return vs
