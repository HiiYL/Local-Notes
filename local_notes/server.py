from __future__ import annotations

import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from .service import search_index, ask_question
from .service import stream_answer, stream_answer_with_history
from .storage.conversations import ConversationDB

app = FastAPI(title="Local Notes API", version="0.1.0")


DEFAULT_STORE = os.environ.get("LOCAL_NOTES_STORE", "./data/index")
DEFAULT_EMBED_MODEL = os.environ.get(
    "LOCAL_NOTES_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

CONV_DB_PATH = os.environ.get("LOCAL_NOTES_CONV_DB", "./data/conversations.db")
conv_db = ConversationDB(CONV_DB_PATH)


WEB_DIR = os.path.join(os.path.dirname(__file__), "web")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
def root():
    index_path = os.path.join(WEB_DIR, "index.html")
    return FileResponse(index_path)


class SearchResult(BaseModel):
    rank: int
    title: str
    folder: str
    chunk: int
    score: float
    text: str


class AskRequest(BaseModel):
    question: str
    k: int = 6
    provider: str = "ollama"
    llm_model: Optional[str] = None
    embed_model: str = DEFAULT_EMBED_MODEL
    store_dir: str = DEFAULT_STORE


class AskResponse(BaseModel):
    answer: str
    sources: List[SearchResult]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/search", response_model=List[SearchResult])
def search(
    q: str = Query(..., min_length=1),
    k: int = Query(5, ge=1, le=20),
    store_dir: str = Query(DEFAULT_STORE),
    embed_model: str = Query(DEFAULT_EMBED_MODEL),
    max_chars: int = Query(300, ge=50, le=2000),
):
    try:
        results = search_index(q, store_dir=store_dir, embed_model=embed_model, k=k, max_chars=max_chars)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return [
        SearchResult(
            rank=r["rank"],
            title=r["title"],
            folder=r["folder"],
            chunk=r["chunk"],
            score=r["score"],
            text=r["text"],
        )
        for r in results
    ]


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        answer_text, sources = ask_question(
            question=req.question,
            store_dir=req.store_dir,
            embed_model=req.embed_model,
            k=req.k,
            provider=req.provider,
            llm_model=req.llm_model,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Truncate source texts in response for payload size
    trimmed: List[SearchResult] = []
    for s in sources:
        t = s["text"]
        if len(t) > 800:
            t = t[:800] + "…"
        trimmed.append(
            SearchResult(
                rank=s["rank"],
                title=s["title"],
                folder=s["folder"],
                chunk=s["chunk"],
                score=0.0,
                text=t,
            )
        )

    return AskResponse(answer=answer_text, sources=trimmed)


@app.post("/ask/stream")
def ask_stream(req: AskRequest):
    def event_gen():
        for ev, data in stream_answer(
            question=req.question,
            store_dir=req.store_dir,
            embed_model=req.embed_model,
            k=req.k,
            provider=req.provider,
            llm_model=req.llm_model,
        ):
            # Server-Sent Events format
            yield f"event: {ev}\n"
            # data may be multiline; SSE supports repeating data: lines, but we keep it single-line JSON/text
            yield f"data: {data}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# --- Conversation models & endpoints ---
class Conversation(BaseModel):
    id: str
    title: str
    created_at: int
    updated_at: int


class Message(BaseModel):
    id: int
    role: str
    content: str
    created_at: int
    citations: Optional[str] = None


class CreateConversationRequest(BaseModel):
    id: str
    title: str = "New Conversation"


class AddMessageRequest(BaseModel):
    role: str
    content: str


@app.get("/conv", response_model=List[Conversation])
def list_conversations():
    return [Conversation(**c) for c in conv_db.list_conversations()]


@app.post("/conv", response_model=Conversation)
def create_conversation(req: CreateConversationRequest):
    import time
    ts = int(time.time())
    conv_db.create_conversation(req.id, req.title, ts)
    return Conversation(id=req.id, title=req.title, created_at=ts, updated_at=ts)


@app.get("/conv/{conv_id}/messages", response_model=List[Message])
def get_messages(conv_id: str, limit: int = 50):
    msgs = conv_db.get_messages(conv_id, limit=limit)
    return [Message(**m) for m in msgs]


@app.post("/conv/{conv_id}/message", response_model=Message)
def add_message(conv_id: str, req: AddMessageRequest):
    import time
    ts = int(time.time())
    mid = conv_db.add_message(conv_id, req.role, req.content, ts)
    return Message(id=mid, role=req.role, content=req.content, created_at=ts)


class ConvAskRequest(AskRequest):
    pass


@app.post("/conv/{conv_id}/ask/stream")
def conv_ask_stream(conv_id: str, req: ConvAskRequest):
    import time

    # Save user message first
    ts = int(time.time())
    conv_db.add_message(conv_id, "user", req.question, ts)
    # Auto-rename conversation to first question if default title
    conv = conv_db.get_conversation(conv_id)
    if conv and (conv.get("title") or "").strip() in ("", "New Conversation"):
        title = req.question.strip()
        if len(title) > 60:
            title = title[:60] + "…"
        conv_db.update_title(conv_id, title, ts)

    def event_gen():
        # Build recent history for better coherence
        history = conv_db.get_messages(conv_id, limit=50)
        # stream answer and at the end save assistant message
        collected = []
        citations_json: Optional[str] = None
        for ev, data in stream_answer_with_history(
            question=req.question,
            history=history,
            store_dir=req.store_dir,
            embed_model=req.embed_model,
            k=req.k,
            provider=req.provider,
            llm_model=req.llm_model,
        ):
            if ev == "delta":
                collected.append(data)
                yield f"event: {ev}\n"
                yield f"data: {data}\n\n"
            elif ev == "sources":
                # pass through sources immediately
                yield f"event: {ev}\n"
                yield f"data: {data}\n\n"
            elif ev == "citations":
                # cited-only sources
                citations_json = data
                yield f"event: {ev}\n"
                yield f"data: {data}\n\n"
            elif ev == "done":
                # persist assistant message BEFORE notifying client 'done'
                full = "".join(collected)
                conv_db.add_message(conv_id, "assistant", full, int(time.time()), citations_json=citations_json)
                yield f"event: done\n"
                yield f"data: \n\n"
        # generator end

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.get("/conv/{conv_id}/export")
def export_conversation(conv_id: str):
    conv = conv_db.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    msgs = conv_db.get_messages(conv_id, limit=10000)
    return {"conversation": conv, "messages": msgs}


class ImportConversationRequest(BaseModel):
    id: str
    title: str
    messages: List[Message]


@app.post("/conv/import", response_model=Conversation)
def import_conversation(req: ImportConversationRequest):
    import time
    ts = int(time.time())
    conv_db.delete_conversation(req.id)
    conv_db.create_conversation(req.id, req.title, ts)
    # Insert messages preserving timestamps if provided
    for m in req.messages:
        conv_db.insert_message(req.id, m.role, m.content, getattr(m, "created_at", ts) or ts)
    return Conversation(id=req.id, title=req.title, created_at=ts, updated_at=ts)


@app.delete("/conv/{conv_id}")
def delete_conversation(conv_id: str):
    conv_db.delete_conversation(conv_id)
    return {"ok": True}
