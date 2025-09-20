from __future__ import annotations

import os
import json
from typing import List, Tuple, Dict, Any, Optional, Iterable
import re

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from .llm.providers import get_llm
from .indexing import whoosh_index


# A concise, instruction-tuned system prompt optimized for Gemma 2 in a local RAG setting
SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions using ONLY the provided note excerpts.\n"
    "Follow these rules strictly:\n"
    "- Cite sources using bracket numbers like [1], [2] that refer to the provided snippets.\n"
    "- Do not invent sources or facts not present in the snippets. If the answer is not in the snippets, say you don't know.\n"
    "- Prefer detailed, well-structured answers. Use short paragraphs AND bullet points when appropriate.\n"
    "- If the user asks for steps or a plan, return a clear, numbered list.\n"
    "- Preserve code blocks and formatting when relevant.\n"
    "- When quoting directly, keep the quote minimal and include a citation, e.g., \"...\" [3].\n"
    "- If multiple snippets conflict, note the disagreement and cite each.\n"
    "- Final section: add a one-line summary and then a 'Sources' line listing the cited numbers in order (e.g., Sources: [2], [5])."
)


def ensure_index_exists(store_dir: str) -> None:
    index_files = ["index.faiss", "index.pkl"]
    if not all(os.path.exists(os.path.join(store_dir, f)) for f in index_files):
        raise FileNotFoundError(f"No FAISS index found in {store_dir}. Run indexing first.")


def load_vectorstore(store_dir: str, embed_model: str) -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=embed_model)
    vs = FAISS.load_local(store_dir, embeddings, allow_dangerous_deserialization=True)
    return vs


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\w+", (text or "").lower()))


def _lexical_score(text: str, query: str) -> float:
    """Lightweight lexical score: try RapidFuzz; fallback to Jaccard overlap."""
    try:
        from rapidfuzz import fuzz  # type: ignore

        return float(fuzz.token_set_ratio(query or "", text or "")) / 100.0
    except Exception:
        q = _tokenize(query)
        t = _tokenize(text)
        if not q or not t:
            return 0.0
        inter = len(q & t)
        union = len(q | t)
        return inter / union


def _rrf_merge(vector_ranked: list, lexical_ranked: list, key_fn) -> list:
    """Reciprocal Rank Fusion merge of two rankings.

    key_fn extracts a stable key per item. Items are the original doc objects.
    """
    K = 60  # rrf constant
    scores: dict[str, float] = {}
    for idx, item in enumerate(vector_ranked, start=1):
        scores[key_fn(item)] = scores.get(key_fn(item), 0.0) + 1.0 / (K + idx)
    for idx, item in enumerate(lexical_ranked, start=1):
        scores[key_fn(item)] = scores.get(key_fn(item), 0.0) + 1.0 / (K + idx)
    # stable order: by fused score desc, then by vector rank
    vec_pos = {key_fn(it): i for i, it in enumerate(vector_ranked)}
    merged = sorted(vector_ranked, key=lambda it: (-scores.get(key_fn(it), 0.0), vec_pos.get(key_fn(it), 1e9)))
    # Append any items only in lexical list (not present in vector list)
    vec_keys = set(vec_pos.keys())
    for it in lexical_ranked:
        k = key_fn(it)
        if k not in vec_keys:
            merged.append(it)
    return merged


def _hybrid_retrieve(vs: FAISS, store_dir: str, query: str, k: int, fetch_k: int, recency_alpha: float = 0.1) -> list:
    """Hybrid retrieval using FAISS (vector) + Whoosh (lexical) fused via RRF, with small recency bias."""
    # Vector pool
    pool = vs.similarity_search_with_score(query, k=fetch_k)
    vec_docs = [d for d, _ in pool]

    # Lexical pool from Whoosh, map to FAISS docstore docs by doc_key (doc_id::chunk_id)
    wres = whoosh_index.search(store_dir, query, fetch_k) or []
    lex_docs = []
    updated_map: Dict[str, int] = {}
    for r in wres:
        dk = r.get("doc_key")
        if dk and dk in getattr(vs.docstore, "_dict", {}):
            doc = vs.docstore._dict[dk]
            lex_docs.append(doc)
            updated_map[dk] = int(r.get("updated_at") or 0)
    # Fallback: if Whoosh empty, degrade to simple lexical over vector pool
    if not lex_docs:
        lex_docs = sorted(vec_docs, key=lambda d: _lexical_score(d.page_content or "", query), reverse=True)

    # Fuse
    key_fn = lambda d: f"{d.metadata.get('doc_id','')}::{int(d.metadata.get('chunk_id', d.metadata.get('chunk', 0)))}"
    fused = _rrf_merge(vec_docs, lex_docs, key_fn)

    # Recency bias: promote docs that have higher updated_at in the Whoosh set
    if updated_map and recency_alpha > 0:
        times = [updated_map.get(key_fn(d), 0) for d in fused]
        if any(times):
            tmax = max(times)
            tmin = min(t for t in times if t)
            span = max(1, tmax - tmin)
            fused = sorted(
                fused,
                key=lambda d: -recency_alpha * ((updated_map.get(key_fn(d), 0) - tmin) / span if updated_map.get(key_fn(d), 0) else 0.0),
            )
    return fused[:k]


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
    # Hybrid retrieval: FAISS + Whoosh with RRF
    fetch_k = max(20, k * 8)
    pool = vs.similarity_search_with_score(query, k=fetch_k)
    score_map = {d.page_content: float(s) for d, s in pool}
    recency_alpha = float(os.environ.get("LOCAL_NOTES_RECENCY_ALPHA", "0.1") or 0.1)
    docs_ranked = _hybrid_retrieve(vs, store_dir, query, k=k, fetch_k=fetch_k, recency_alpha=recency_alpha)
    out: List[Dict[str, Any]] = []
    for rank, doc in enumerate(docs_ranked, start=1):
        score = score_map.get(doc.page_content, 0.0)
        m = doc.metadata or {}
        content = doc.page_content or ""
        snippet = content[:max_chars] + ("…" if len(content) > max_chars else "")
        out.append(
            {
                "rank": rank,
                "title": m.get("title", ""),
                "folder": str(m.get("folder", "")),
                "chunk": int(m.get("chunk", 0)),
                "doc_id": m.get("doc_id", ""),
                "chunk_id": int(m.get("chunk_id", m.get("chunk", 0))),
                "heading": m.get("heading", ""),
                "updated_at": m.get("updated_at", ""),
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
    fetch_k = max(20, k * 8)
    alpha = recency_alpha if recency_alpha is not None else float(os.environ.get("LOCAL_NOTES_RECENCY_ALPHA", "0.1") or 0.1)
    docs = _hybrid_retrieve(vs, store_dir, question, k=k, fetch_k=fetch_k, recency_alpha=alpha)
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
            "doc_id": m.get("doc_id", ""),
            "chunk_id": int(m.get("chunk_id", m.get("chunk", 0))),
            "heading": m.get("heading", ""),
            "updated_at": m.get("updated_at", ""),
            "score": 0.0,
            "text": d.page_content or "",
        }
        sources.append(src)
        blocks.append(f"[{i}] Title: {src['title']}\nFolder: {src['folder']}\nChunk: {src['chunk']}\n---\n{src['text']}")

    system = SYSTEM_PROMPT
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
    recency_alpha: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """Retrieve docs and build the user prompt. Returns (sources, prompt)."""
    ensure_index_exists(store_dir)
    vs = load_vectorstore(store_dir, embed_model)
    fetch_k = max(20, k * 8)
    recency_alpha = float(os.environ.get("LOCAL_NOTES_RECENCY_ALPHA", "0.1") or 0.1)
    docs = _hybrid_retrieve(vs, store_dir, question, k=k, fetch_k=fetch_k, recency_alpha=recency_alpha)
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
    recency_alpha: Optional[float] = None,
) -> Iterable[Tuple[str, str]]:
    """Yield a stream of (event, data) tuples.

    Events:
    - ("sources", json_string)
    - ("delta", text_chunk)
    - ("done", "")
    """
    sources, user_prompt = _prepare_sources_and_prompt(question, store_dir, embed_model, k, recency_alpha=recency_alpha)
    if not sources:
        yield ("sources", json.dumps([]))
        yield ("delta", "I couldn't find anything relevant in your notes.")
        yield ("done", "")


def stream_answer_with_history(
    question: str,
    history: List[Dict[str, str]],  # [{"role": "user"|"assistant", "content": str}]
    store_dir: str,
    embed_model: str,
    k: int = 6,
    provider: str = "ollama",
    llm_model: Optional[str] = None,
    recency_alpha: float = 0.1,
) -> Iterable[Tuple[str, str]]:
    """Stream answer but include recent chat history for better coherence.

    Retrieval query is lightly augmented with last few user/assistant turns.
    The generated answer is still constrained to the provided note snippets.
    """
    # Build a compact history prefix for disambiguation only (not as a source)
    # - keep last up to 6 turns
    # - truncate each to 200 chars
    hist = history[-6:] if history else []
    history_lines = []
    for m in hist:
        role = (m.get("role", "user")).capitalize()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if len(content) > 200:
            content = content[:200] + "…"
        history_lines.append(f"{role}: {content}")
    history_text = "\n".join(history_lines)

    # Retrieval MUST use only the current question to avoid history leakage
    aug_query = question

    # Retrieval and prompt prep
    ensure_index_exists(store_dir)
    vs = load_vectorstore(store_dir, embed_model)
    fetch_k = max(20, k * 8)
    docs = _hybrid_retrieve(vs, store_dir, aug_query, k=k, fetch_k=fetch_k, recency_alpha=recency_alpha)
    if not docs:
        yield ("sources", json.dumps([]))
        yield ("delta", "I couldn't find anything relevant in your notes.")
        yield ("done", "")
        return

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
        (f"Conversation context (for intent only; not a source to cite):\n{history_text}\n\n" if history_text else "") +
        f"Question: {question}\n\nContext:\n" + "\n\n".join(blocks) +
        "\n\nAnswer concisely and cite sources like [1], [2] where appropriate. Do not cite the conversation context."
    )

    system = SYSTEM_PROMPT

    llm = get_llm(provider=provider, model=llm_model)
    yield ("sources", json.dumps(sources))
    full_text_parts: List[str] = []
    emitted_citations: set[int] = set()
    for chunk in llm.stream([
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt},
    ]):
        part = getattr(chunk, "content", str(chunk))
        if part:
            full_text_parts.append(part)
            yield ("delta", part)
            partial = "".join(full_text_parts)
            cited_now = set(int(m.group(1)) for m in re.finditer(r"\[(\d+)\]", partial))
            if cited_now - emitted_citations:
                emitted_citations = cited_now
                cited_sources = [s for s in sources if s.get("rank") in cited_now]
                yield ("citations", json.dumps(cited_sources))
    full_text = "".join(full_text_parts)
    cited = set(int(m.group(1)) for m in re.finditer(r"\[(\d+)\]", full_text))
    if cited:
        cited_sources = [s for s in sources if s.get("rank") in cited]
        yield ("citations", json.dumps(cited_sources))
    yield ("done", "")
