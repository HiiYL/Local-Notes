from __future__ import annotations

import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from .service import search_index, ask_question
from .service import stream_answer

app = FastAPI(title="Local Notes API", version="0.1.0")


DEFAULT_STORE = os.environ.get("LOCAL_NOTES_STORE", "./data/index")
DEFAULT_EMBED_MODEL = os.environ.get(
    "LOCAL_NOTES_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)


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
