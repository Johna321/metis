from __future__ import annotations

import json
import logging
import queue
import threading
from collections.abc import Iterable
from enum import Enum
from typing import List, Optional, Tuple

import orjson
import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel

from ..core.agent import run_agent
from ..core.ingest import ingest_pdf_bytes, ingest_pdf_bytes_layout
from ..core.llm import AnthropicModel, OpenAIModel, OpenRouterModel, StreamEvent
from ..core.prompts import SYSTEM_PROMPT
from ..core.retrieve import retrieve
from ..core.store import paths
from ..core.tools import ToolRegistry, make_rag_retrieve_tool, make_web_search_tool
from ..core.vectorize import retrieve_semantic, vectorize_spans
from ..settings import (
    AGENT_MAX_ITER,
    AGENT_TEMPERATURE,
    ANTHROPIC_API_KEY,
    LLM_API_KEY,
    LLM_MODEL,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENROUTER_API_KEY,
    TAVILY_API_KEY,
)

app = FastAPI(title="Metis")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:1420",
        "tauri://localhost",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class Engine(str, Enum):
    blocks = "blocks"
    layout = "layout"


class IngestResponse(BaseModel):
    doc_id: str
    n_pages: int
    n_spans: int
    ingest: dict


class RetrieveRequest(BaseModel):
    doc_id: str
    page: int
    selected_text: str


class VectorizeRequest(BaseModel):
    doc_id: str


class VectorizeResponse(BaseModel):
    doc_id: str
    n_embedded: int
    n_skipped: Optional[int] = None
    model: str
    dim: Optional[int] = None


class SemanticRetrieveRequest(BaseModel):
    doc_id: str
    query: str
    page: Optional[int] = None
    top_k: Optional[int] = None


class EvidenceItem(BaseModel):
    span_id: str
    page: int
    bbox_norm: Tuple[float, float, float, float]
    text: str
    score: float


class ChatRequest(BaseModel):
    doc_id: str
    message: str
    provider: Optional[str] = None
    model: Optional[str] = None


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def _stream_event_to_sse(event: StreamEvent) -> ServerSentEvent | None:
    if event.kind == "text_delta":
        return ServerSentEvent(data={"text": event.text}, event="text_delta")
    elif event.kind == "tool_call_start":
        return ServerSentEvent(data={"name": event.text}, event="tool_call_start")
    elif event.kind == "tool_call_delta":
        return ServerSentEvent(data={"text": event.text}, event="tool_call_delta")
    elif event.kind == "tool_call_done" and event.tool_call:
        return ServerSentEvent(
            data={
                "id": event.tool_call.id,
                "name": event.tool_call.name,
                "arguments": event.tool_call.arguments,
            },
            event="tool_call_done",
        )
    elif event.kind == "message_done" and event.message:
        msg = event.message
        payload: dict = {"role": msg.role, "content": msg.content}
        if msg.tool_calls:
            payload["tool_calls"] = [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                for tc in msg.tool_calls
            ]
        return ServerSentEvent(data=payload, event="message_done")
    return None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(
    file: UploadFile = File(...),
    engine: Engine = Query(Engine.blocks),
    extract_words: bool = Query(False),
    write_images: bool = Query(False),
    dpi: int = Query(200),
):
    pdf_bytes = await file.read()
    if engine == Engine.layout:
        meta = ingest_pdf_bytes_layout(
            pdf_bytes,
            extract_words=extract_words,
            write_images=write_images,
            dpi=dpi,
        )
    else:
        meta = ingest_pdf_bytes(pdf_bytes)
    return meta


@app.post("/retrieve", response_model=List[EvidenceItem])
async def retrieve_endpoint(req: RetrieveRequest):
    p = paths(req.doc_id)
    if not p["spans"].exists():
        raise HTTPException(status_code=404, detail=f"Document not found: {req.doc_id}")
    evidence = retrieve(doc_id=req.doc_id, page=req.page, selected_text=req.selected_text)
    return [e.__dict__ for e in evidence]


@app.post("/vectorize", response_model=VectorizeResponse)
def vectorize_endpoint(req: VectorizeRequest):
    p = paths(req.doc_id)
    if not p["spans"].exists():
        raise HTTPException(status_code=404, detail=f"Document not found: {req.doc_id}")
    result = vectorize_spans(doc_id=req.doc_id)
    return result


@app.post("/retrieve-semantic", response_model=List[EvidenceItem])
def retrieve_semantic_endpoint(req: SemanticRetrieveRequest):
    kwargs = {}
    if req.page is not None:
        kwargs["page"] = req.page
    if req.top_k is not None:
        kwargs["top_k"] = req.top_k
    try:
        evidence = retrieve_semantic(doc_id=req.doc_id, query=req.query, **kwargs)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Embeddings not found for this document. Run 'metis vectorize <doc_id>' first.",
        )
    return [
        EvidenceItem(span_id=e.span_id, page=e.page, bbox_norm=e.bbox_norm, text=e.text, score=e.score)
        for e in evidence
    ]


@app.get("/documents/{doc_id}")
async def get_document(doc_id: str):
    p = paths(doc_id)
    if not p["doc"].exists():
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")
    return orjson.loads(p["doc"].read_bytes())


@app.get("/documents/{doc_id}/pdf")
async def get_document_pdf(doc_id: str):
    p = paths(doc_id)
    if not p["pdf"].exists():
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")
    return FileResponse(p["pdf"], media_type="application/pdf")


@app.post("/chat", response_class=EventSourceResponse)
def chat_endpoint(req: ChatRequest) -> Iterable[ServerSentEvent]:
    # Validate document exists
    p = paths(req.doc_id)
    if not p["spans"].exists():
        raise HTTPException(status_code=404, detail=f"Document not found: {req.doc_id}")
    if not p["embeddings"].exists():
        raise HTTPException(
            status_code=400,
            detail=f"No embeddings found. Run 'metis vectorize {req.doc_id}' first.",
        )

    # Resolve provider / model / api_key
    prov = req.provider or LLM_PROVIDER
    mod = req.model or LLM_MODEL
    api_key = LLM_API_KEY
    if not api_key:
        if prov == "anthropic":
            api_key = ANTHROPIC_API_KEY
        elif prov == "openai":
            api_key = OPENAI_API_KEY
        elif prov == "openrouter":
            api_key = OPENROUTER_API_KEY
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="No API key configured. Set METIS_LLM_API_KEY or ANTHROPIC_API_KEY / OPENAI_API_KEY / OPENROUTER_API_KEY.",
        )

    if prov == "anthropic":
        llm = AnthropicModel(api_key=api_key, model=mod, temperature=AGENT_TEMPERATURE)
    elif prov == "openai":
        llm = OpenAIModel(api_key=api_key, model=mod, temperature=AGENT_TEMPERATURE)
    elif prov == "openrouter":
        llm = OpenRouterModel(api_key=api_key, model=mod, temperature=AGENT_TEMPERATURE)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {prov}. Use 'anthropic', 'openai', or 'openrouter'.")

    # Build tools
    registry = ToolRegistry()
    rag_def, rag_fn = make_rag_retrieve_tool(req.doc_id)
    registry.register(rag_def.name, rag_def.description, rag_def.parameters, rag_fn)

    tavily_key = TAVILY_API_KEY
    if tavily_key:
        ws_def, ws_fn = make_web_search_tool(tavily_key)
        registry.register(ws_def.name, ws_def.description, ws_def.parameters, ws_fn)

    # Queue-bridged SSE generator: run_agent uses a callback, not a generator,
    # so we bridge it via a queue consumed by a sync generator (Starlette runs
    # sync generators in a threadpool automatically).
    q: queue.Queue[StreamEvent | Exception | None] = queue.Queue()

    def on_stream(event: StreamEvent) -> None:
        q.put(event)

    def agent_thread() -> None:
        try:
            run_agent(
                model=llm,
                user_query=req.message,
                tools=registry,
                system_prompt=SYSTEM_PROMPT,
                max_iterations=AGENT_MAX_ITER,
                on_stream=on_stream,
            )
        except Exception as exc:
            q.put(exc)
        finally:
            q.put(None)  # sentinel

    def sse_generator() -> Iterable[ServerSentEvent]:
        thread = threading.Thread(target=agent_thread, daemon=True)
        thread.start()
        while True:
            item = q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                yield ServerSentEvent(
                    data={"detail": str(item)}, event="error"
                )
                break
            sse = _stream_event_to_sse(item)
            if sse is not None:
                yield sse

    return sse_generator()  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    uvicorn.run(app, host="0.0.0.0", port=8000)
