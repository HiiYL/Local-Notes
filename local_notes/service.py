from __future__ import annotations

import os
import json
from typing import List, Tuple, Dict, Any, Optional, Iterable

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from .llm.providers import get_llm


def ensure_index_exists(store_dir: str) -> None:
    index_files = ["index.faiss", "index.pkl"]
    if not all(os.path.exists(os.path.join(store_dir, f)) for f in index_files):
        raise FileNotFoundError(f"No FAISS index found in {store_dir}. Run indexing first.")


def load_vectorstore(store_dir: str, embed_model: str) -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=embed_model)
    vs = FAISS.load_local(store_dir, embeddings, allow_dangerous_deserialization=True)
    return vs


def search_index(
    query: str,
    store_dir: str,
    embed_model: str,
    k: int = 5,
    max_chars: int = 300,
) -> List[Dict[str, Any]]:
    """Search the FAISS index and return list of dicts with metadata and snippet."""
    ensure_index_exists(store_dir)
    vs = load_vectorstore(store_dir, embed_model)
    docs_and_scores = vs.similarity_search_with_score(query, k=k)
    out: List[Dict[str, Any]] = []
    for rank, (doc, score) in enumerate(docs_and_scores, start=1):
        m = doc.metadata or {}
        content = doc.page_content or ""
        snippet = content[:max_chars] + ("…" if len(content) > max_chars else "")
        out.append(
            {
                "rank": rank,
                "title": m.get("title", ""),
                "folder": str(m.get("folder", "")),
                "chunk": int(m.get("chunk", 0)),
                "score": float(score),
                "text": snippet,
                "raw": content,
            }
        )
    return out


def ask_question(
    question: str,
    store_dir: str,
    embed_model: str,
    k: int = 6,
    provider: str = "ollama",
    llm_model: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Retrieve top-k, build prompt, and get an LLM answer. Returns (answer, sources)."""
    ensure_index_exists(store_dir)
    vs = load_vectorstore(store_dir, embed_model)
    docs = vs.similarity_search(question, k=k)
    if not docs:
        return ("I couldn't find anything relevant in your notes.", [])

    blocks: List[str] = []
    sources: List[Dict[str, Any]] = []
    for i, d in enumerate(docs, start=1):
        m = d.metadata or {}
        src = {
            "rank": i,
            "title": m.get("title", "Untitled"),
            "folder": str(m.get("folder", "")),
            "chunk": int(m.get("chunk", 0)),
            "score": 0.0,
            "text": d.page_content or "",
        }
        sources.append(src)
        blocks.append(f"[{i}] Title: {src['title']}\nFolder: {src['folder']}\nChunk: {src['chunk']}\n---\n{src['text']}")

    system = (
        "You are a helpful assistant answering questions using ONLY the provided note excerpts. "
        "Cite sources using bracket numbers like [1], [2] that refer to the provided snippets. "
        "If the answer is not contained in the snippets, say you don't know."
    )
    user_prompt = (
        f"Question: {question}\n\nContext:\n" + "\n\n".join(blocks) +
        "\n\nAnswer concisely and cite sources like [1], [2] where appropriate."
    )

    llm = get_llm(provider=provider, model=llm_model)
    resp = llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": user_prompt}])
    answer_text = getattr(resp, "content", str(resp))
    return answer_text, sources


def _prepare_sources_and_prompt(
    question: str,
    store_dir: str,
    embed_model: str,
    k: int,
) -> Tuple[List[Dict[str, Any]], str]:
    """Retrieve docs and build the user prompt. Returns (sources, prompt)."""
    ensure_index_exists(store_dir)
    vs = load_vectorstore(store_dir, embed_model)
    docs = vs.similarity_search(question, k=k)
    if not docs:
        return [], ""
    blocks: List[str] = []
    sources: List[Dict[str, Any]] = []
    for i, d in enumerate(docs, start=1):
        m = d.metadata or {}
        src = {
            "rank": i,
            "title": m.get("title", "Untitled"),
            "folder": str(m.get("folder", "")),
            "chunk": int(m.get("chunk", 0)),
            "score": 0.0,
            "text": d.page_content or "",
        }
        sources.append(src)
        blocks.append(f"[{i}] Title: {src['title']}\nFolder: {src['folder']}\nChunk: {src['chunk']}\n---\n{src['text']}")
    user_prompt = (
        f"Question: {question}\n\nContext:\n" + "\n\n".join(blocks) +
        "\n\nAnswer concisely and cite sources like [1], [2] where appropriate."
    )
    return sources, user_prompt


def stream_answer(
    question: str,
    store_dir: str,
    embed_model: str,
    k: int = 6,
    provider: str = "ollama",
    llm_model: Optional[str] = None,
) -> Iterable[Tuple[str, str]]:
    """Yield a stream of (event, data) tuples.

    Events:
    - ("sources", json_string)
    - ("delta", text_chunk)
    - ("done", "")
    """
    sources, user_prompt = _prepare_sources_and_prompt(question, store_dir, embed_model, k)
    if not sources:
        yield ("sources", json.dumps([]))
        yield ("delta", "I couldn't find anything relevant in your notes.")
        yield ("done", "")
        return

    system = (
        "You are a helpful assistant answering questions using ONLY the provided note excerpts. "
        "Cite sources using bracket numbers like [1], [2] that refer to the provided snippets. "
        "If the answer is not contained in the snippets, say you don't know."
    )

    llm = get_llm(provider=provider, model=llm_model)
    # Emit sources first
    yield ("sources", json.dumps(sources))
    # Stream deltas
    for chunk in llm.stream([
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt},
    ]):
        part = getattr(chunk, "content", str(chunk))
        if part:
            yield ("delta", part)
    yield ("done", "")
