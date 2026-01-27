from __future__ import annotations
import pymupdf
from typing import List, Tuple
from .schema import Span
from .store import doc_id_from_bytes, paths, write_json, write_spans_jsonl
from ..settings import MIN_CHARS

def _norm_bbox(b: Tuple[float,float,float,float], w: float, h: float):
    x0,y0,x1,y1 = b
    return (x0/w, y0/h, x1/w, y1/h)

def ingest_pdf_bytes(pdf_bytes: bytes) -> dict:
    doc_id = doc_id_from_bytes(pdf_bytes)
    p = paths(doc_id)
    p["pdf"].write_bytes(pdf_bytes)

    d = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    spans: List[Span] = []
    ro = 0

    for page_i in range(d.page_count):
        page = d[page_i]
        w, h = page.rect.width, page.rect.height

        # blocks: (x0,y0,x1,y1,"text", block_no, block_type)
        blocks = page.get_text("blocks")
        # sort by top-left reading order (good enough v0)
        blocks.sort(key=lambda b: (b[1], b[0]))

        for bi, b in enumerate(blocks):
            x0,y0,x1,y1,text,*_ = b
            t = " ".join(text.split())
            if len(t) < MIN_CHARS:
                continue
            span = Span(
                span_id=f"p{page_i:03d}_b{bi:03d}",
                doc_id=doc_id,
                page=page_i,
                bbox_pdf=(float(x0),float(y0),float(x1),float(y1)),
                bbox_norm=_norm_bbox((float(x0),float(y0),float(x1),float(y1)), w, h),
                text=t,
                reading_order=ro,
                is_header=False,
                is_footer=False,
            )
            spans.append(span)
            ro += 1

    meta = {
        "doc_id": doc_id,
        "n_pages": d.page_count,
        "n_spans": len(spans),
        "ingest": {"engine": "pymupdf", "min_chars": MIN_CHARS},
    }
    write_json(p["doc"], meta)
    write_spans_jsonl(p["spans"], spans)
    return meta
