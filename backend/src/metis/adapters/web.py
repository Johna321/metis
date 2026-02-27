from __future__ import annotations

import logging
from enum import Enum
from typing import List, Optional, Tuple

import orjson
import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from ..core.ingest import ingest_pdf_bytes, ingest_pdf_bytes_layout
from ..core.retrieve import retrieve
from ..core.store import paths
from ..core.vectorize import retrieve_semantic, vectorize_spans

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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    uvicorn.run(app, host="0.0.0.0", port=8000)
