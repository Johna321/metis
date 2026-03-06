"""FastAPI annotation server for gold-standard PDF annotation."""

from __future__ import annotations

import uuid
from functools import lru_cache
from pathlib import Path

import pymupdf
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

from ..benchmark.gold import (
    bootstrap_from_spans,
    export_to_gold,
    load_annotation_state,
    save_annotation_state,
)
from ..core.store import paths

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


# ---------- request/response models ----------

class SpanUpdate(BaseModel):
    bbox_norm: list[float] | None = None
    bbox_pdf: list[float] | None = None
    kind: str | None = None
    status: str | None = None
    group_id: str | None = None
    reading_order: int | None = None
    text: str | None = None


class SpanCreate(BaseModel):
    page: int
    bbox_norm: list[float]
    bbox_pdf: list[float]
    kind: str = "text"
    text: str = ""
    reading_order: int = 0
    group_id: str | None = None


class GroupRequest(BaseModel):
    ann_ids: list[str]
    group_id: str | None = None


# ---------- app factory ----------

def create_app(doc_id: str, dpi: int = 150) -> FastAPI:
    app = FastAPI(title="Metis Annotation Tool")
    p = paths(doc_id)

    if not p["pdf"].exists():
        raise FileNotFoundError(f"PDF not found for {doc_id}")
    if not p["spans"].exists():
        raise FileNotFoundError(f"Spans not found for {doc_id} — ingest the PDF first")

    # Load or bootstrap state
    state = load_annotation_state(doc_id)
    if state is None:
        state = bootstrap_from_spans(doc_id)
        save_annotation_state(doc_id, state)

    # Get page count
    pdf_doc = pymupdf.open(p["pdf"])
    page_count = len(pdf_doc)
    pdf_doc.close()

    # --- page image cache ---
    @lru_cache(maxsize=32)
    def render_page(page_num: int) -> bytes:
        doc = pymupdf.open(p["pdf"])
        page = doc[page_num]
        pix = page.get_pixmap(dpi=dpi)
        data = pix.tobytes("png")
        doc.close()
        return data

    # --- helpers ---
    def _find_span(ann_id: str) -> tuple[str, int, dict] | None:
        for page_key, anns in state["pages"].items():
            for i, a in enumerate(anns):
                if a["ann_id"] == ann_id:
                    return page_key, i, a
        return None

    def _save():
        save_annotation_state(doc_id, state)

    # --- routes ---

    @app.get("/", response_class=HTMLResponse)
    async def index():
        html_path = STATIC_DIR / "annotate.html"
        return HTMLResponse(html_path.read_text())

    @app.get("/api/doc")
    async def doc_meta():
        return {"doc_id": doc_id, "page_count": page_count, "dpi": dpi}

    @app.get("/api/page/{page_num}/image")
    async def page_image(page_num: int):
        if page_num < 0 or page_num >= page_count:
            raise HTTPException(404, "Page not found")
        data = render_page(page_num)
        return Response(content=data, media_type="image/png")

    @app.get("/api/page/{page_num}/spans")
    async def page_spans(page_num: int):
        key = str(page_num)
        anns = state["pages"].get(key, [])
        # Filter out soft-deleted
        visible = [a for a in anns if a.get("status") != "deleted"]
        return visible

    @app.put("/api/span/{ann_id}")
    async def update_span(ann_id: str, body: SpanUpdate):
        result = _find_span(ann_id)
        if result is None:
            raise HTTPException(404, f"Span {ann_id} not found")
        _, _, span = result
        updates = body.model_dump(exclude_none=True)
        span.update(updates)
        _save()
        return span

    @app.post("/api/span", status_code=201)
    async def create_span(body: SpanCreate):
        page_key = str(body.page)
        ann_id = f"a_manual_{uuid.uuid4().hex[:8]}"
        ann = {
            "ann_id": ann_id,
            "bbox_norm": body.bbox_norm,
            "bbox_pdf": body.bbox_pdf,
            "kind": body.kind,
            "reading_order": body.reading_order,
            "text": body.text,
            "status": "pending",
            "provenance": "manual",
            "group_id": body.group_id,
            "source_span_id": None,
        }
        state["pages"].setdefault(page_key, []).append(ann)
        _save()
        return ann

    @app.delete("/api/span/{ann_id}")
    async def delete_span(ann_id: str):
        result = _find_span(ann_id)
        if result is None:
            raise HTTPException(404, f"Span {ann_id} not found")
        _, _, span = result
        span["status"] = "deleted"
        _save()
        return {"ok": True}

    @app.post("/api/group")
    async def create_group(body: GroupRequest):
        gid = body.group_id or f"grp_{uuid.uuid4().hex[:6]}"
        found = 0
        for ann_id in body.ann_ids:
            result = _find_span(ann_id)
            if result:
                _, _, span = result
                span["group_id"] = gid
                found += 1
        if found == 0:
            raise HTTPException(404, "No matching spans found")
        _save()
        return {"group_id": gid, "linked": found}

    @app.delete("/api/group/{group_id}")
    async def delete_group(group_id: str):
        count = 0
        for anns in state["pages"].values():
            for a in anns:
                if a.get("group_id") == group_id:
                    a["group_id"] = None
                    count += 1
        if count == 0:
            raise HTTPException(404, f"Group {group_id} not found")
        _save()
        return {"unlinked": count}

    @app.post("/api/export")
    async def export():
        out_path = export_to_gold(state)
        # Count accepted
        n_accepted = sum(
            1 for anns in state["pages"].values()
            for a in anns if a["status"] == "accepted"
        )
        return {"path": str(out_path), "n_accepted": n_accepted}

    @app.get("/api/progress")
    async def progress():
        result = {}
        total = {"accepted": 0, "pending": 0, "rejected": 0}
        for page_key, anns in state["pages"].items():
            counts = {"accepted": 0, "pending": 0, "rejected": 0}
            for a in anns:
                s = a.get("status", "pending")
                if s in counts:
                    counts[s] += 1
            result[page_key] = counts
            for k in total:
                total[k] += counts[k]
        result["total"] = total
        return result

    return app
