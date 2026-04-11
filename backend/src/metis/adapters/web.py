from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
from collections.abc import Iterable
from enum import Enum
from typing import List, Optional

import orjson
import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel

from ..core.agent import run_agent
from ..core.generated_types import (
    BboxSelection as BBoxSelection,
    ChatRequest,
    EvidenceItem,
    IngestResponse,
    VectorizeResponse,
)
from ..core.ingest import ingest_pdf_bytes, ingest_pdf_bytes_layout, ingest_pdf_bytes_tree
from ..core.llm import AnthropicModel, OpenAIModel, OpenRouterModel, StreamEvent
from ..core.prompts import SYSTEM_PROMPT, build_system_prompt, format_query_with_selections
from ..core.retrieve import resolve_selections, retrieve
from ..core.store import paths, conv_path, read_conversations, create_conversation, update_conversation, delete_conversation, read_messages, append_message
from ..core.toc import render_toc
from ..core.tools import (
    ToolRegistry,
    make_locate_tool,
    make_rag_retrieve_tool,
    make_read_page_tool,
    make_read_section_tool,
    make_web_search_tool,
)
from ..core.tree_store import read_tree
from ..core.vectorize import retrieve_semantic, vectorize_spans
from .. import settings as _settings

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
    tree = "tree"
    blocks = "blocks"
    layout = "layout"


class RetrieveRequest(BaseModel):
    doc_id: str
    page: int
    selected_text: str


class VectorizeRequest(BaseModel):
    doc_id: str


class SemanticRetrieveRequest(BaseModel):
    doc_id: str
    query: str
    page: Optional[int] = None
    top_k: Optional[int] = None

class ConversationUpdateRequest(BaseModel):
    title: Optional[str] = None
    pinned: Optional[bool] = None

class SettingsUpdate(BaseModel):
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
    elif event.kind == "citation_data" and event.evidence:
        return ServerSentEvent(data={"tool_call_id": event.tool_call_id, "tool_name": event.tool_name, "items": event.evidence}, event="citation_data")
    elif event.kind == "title_update":
        return ServerSentEvent(data={"conv_id": event.tool_call_id, "title": event.text}, event="title_update")
    elif event.kind == "agent_done":
        return ServerSentEvent(data={}, event="agent_done")
    return None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.put("/settings")
def update_settings_endpoint(req: SettingsUpdate) -> dict:
    if req.provider is not None:
        _settings.LLM_PROVIDER = req.provider
    if req.model is not None:
        _settings.LLM_MODEL = req.model
    return {"ok": True}


@app.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(
    file: UploadFile = File(...),
    engine: Engine = Query(Engine.tree),
    parser: Optional[str] = Query(None),
    extract_words: bool = Query(True),
    write_images: bool = Query(True),
    dpi: int = Query(200),
):
    pdf_bytes = await file.read()
    source_filename = file.filename or None
    if engine == Engine.tree:
        meta = await asyncio.to_thread(
            ingest_pdf_bytes_tree,
            pdf_bytes,
            source_filename=source_filename,
            parser_name=parser,
        )
    elif engine == Engine.layout:
        meta = await asyncio.to_thread(
            ingest_pdf_bytes_layout,
            pdf_bytes,
            extract_words=extract_words,
            write_images=write_images,
            dpi=dpi,
            source_filename=source_filename,
        )
    else:
        meta = await asyncio.to_thread(ingest_pdf_bytes, pdf_bytes, source_filename=source_filename)
    return meta


@app.post("/retrieve", response_model=List[EvidenceItem])
def retrieve_endpoint(req: RetrieveRequest):
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
def get_document(doc_id: str):
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


@app.get("/documents/{doc_id}/conversations")
def list_conversations(doc_id: str):
    p = paths(doc_id)
    if not p["doc"].exists():
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")
    return {"conversations": read_conversations(doc_id)}


@app.post("/documents/{doc_id}/conversations")
def create_conversation_endpoint(doc_id: str):
    p = paths(doc_id)
    if not p["doc"].exists():
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")
    return create_conversation(doc_id)


@app.get("/documents/{doc_id}/conversations/{conv_id}")
def get_conversation(doc_id: str, conv_id: str):
    p = paths(doc_id)
    if not p["doc"].exists():
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")
    convs = read_conversations(doc_id)
    meta = next((c for c in convs if c["id"] == conv_id), None)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Conversation not found: {conv_id}")
    messages = read_messages(doc_id, conv_id)
    return {"id": meta["id"], "title": meta["title"], "pinned": meta["pinned"], "messages": messages}


@app.patch("/documents/{doc_id}/conversations/{conv_id}")
def update_conversation_endpoint(doc_id: str, conv_id: str, req: ConversationUpdateRequest):
    try:
        return update_conversation(doc_id, conv_id, title=req.title, pinned=req.pinned)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Conversation not found: {conv_id}")


@app.delete("/documents/{doc_id}/conversations/{conv_id}", status_code=204)
def delete_conversation_endpoint(doc_id: str, conv_id: str):
    delete_conversation(doc_id, conv_id)
    return Response(status_code=204)


@app.post("/chat", response_class=EventSourceResponse)
def chat_endpoint(req: ChatRequest) -> Iterable[ServerSentEvent]:
    # Validate document exists. Accept either the new tree-based pipeline
    # (tree.json + embeddings_v2.npy) or the legacy span pipeline
    # (spans.jsonl + embeddings.npy).
    p = paths(req.doc_id)
    has_tree = p["tree"].exists()
    has_spans = p["spans"].exists()
    if not (has_tree or has_spans):
        raise HTTPException(status_code=404, detail=f"Document not found: {req.doc_id}")
    if has_tree:
        if not p["embeddings_v2"].exists():
            raise HTTPException(
                status_code=400,
                detail=f"No embeddings found. Run the tree embedding pipeline on {req.doc_id} first.",
            )
    else:
        if not p["embeddings"].exists():
            raise HTTPException(
                status_code=400,
                detail=f"No embeddings found. Run 'metis vectorize {req.doc_id}' first.",
            )

    # Resolve provider / model / api_key
    prov = req.provider or _settings.LLM_PROVIDER
    mod = req.model or _settings.LLM_MODEL
    api_key = _settings.LLM_API_KEY
    if not api_key:
        if prov == "anthropic":
            api_key = _settings.ANTHROPIC_API_KEY
        elif prov == "openai":
            api_key = _settings.OPENAI_API_KEY
        elif prov == "openrouter":
            api_key = _settings.OPENROUTER_API_KEY
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="No API key configured. Set METIS_LLM_API_KEY or ANTHROPIC_API_KEY / OPENAI_API_KEY / OPENROUTER_API_KEY.",
        )

    if prov == "anthropic":
        llm = AnthropicModel(api_key=api_key, model=mod, temperature=_settings.AGENT_TEMPERATURE)
    elif prov == "openai":
        llm = OpenAIModel(api_key=api_key, model=mod, temperature=_settings.AGENT_TEMPERATURE)
    elif prov == "openrouter":
        llm = OpenRouterModel(api_key=api_key, model=mod, temperature=_settings.AGENT_TEMPERATURE)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {prov}. Use 'anthropic', 'openai', or 'openrouter'.")

    # Build tools. If a tree sidecar exists we use the hierarchical tools
    # (locate + read_section) and a TOC-aware system prompt. Otherwise we
    # fall back to the legacy span-based RAG tool + flat system prompt.
    registry = ToolRegistry()
    tree_path = p["tree"]
    if tree_path.exists():
        loc_def, loc_fn = make_locate_tool(req.doc_id)
        registry.register(loc_def.name, loc_def.description, loc_def.parameters, loc_fn)

        rs_def, rs_fn = make_read_section_tool(req.doc_id)
        registry.register(rs_def.name, rs_def.description, rs_def.parameters, rs_fn)

        system_prompt = build_system_prompt(render_toc(read_tree(tree_path)))
    else:
        rag_def, rag_fn = make_rag_retrieve_tool(req.doc_id)
        registry.register(rag_def.name, rag_def.description, rag_def.parameters, rag_fn)

        system_prompt = SYSTEM_PROMPT

    if p["page_md"].exists():
        rp_def, rp_fn = make_read_page_tool(req.doc_id)
        registry.register(rp_def.name, rp_def.description, rp_def.parameters, rp_fn)

    tavily_key = _settings.TAVILY_API_KEY
    if tavily_key:
        ws_def, ws_fn = make_web_search_tool(tavily_key)
        registry.register(ws_def.name, ws_def.description, ws_def.parameters, ws_fn)

    # Resolve bbox selections to spans
    enriched_query = req.message
    if req.selections:
        sel_dicts = [{"page": s.page, "bbox_norm": s.bbox_norm} for s in req.selections]
        resolved = resolve_selections(req.doc_id, sel_dicts)
        enriched_query = format_query_with_selections(req.message, resolved)

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
                doc_id=req.doc_id,
                user_query=enriched_query,
                tools=registry,
                system_prompt=system_prompt,
                conv_id=req.conv_id,
                max_iterations=_settings.AGENT_MAX_ITER,
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
                    data={"message": str(item)}, event="error"
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
