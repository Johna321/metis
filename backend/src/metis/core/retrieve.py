from __future__ import annotations
from typing import List
from rapidfuzz import fuzz
from .schema import BBox, Evidence, Span
from .store import paths, read_spans_jsonl
from ..settings import TOPK_EVIDENCE, NEIGHBOR_WINDOW

def retrieve(doc_id: str, page: int, selected_text: str) -> List[Evidence]:
    p = paths(doc_id)
    spans: List[Span] = read_spans_jsonl(p["spans"])
    cand = [s for s in spans if s.page == page and not (s.is_header or s.is_footer)]

    q = " ".join(selected_text.split())
    scored = []
    for s in cand:
        score = fuzz.partial_ratio(q, s.text)
        scored.append((score, s))
    scored.sort(key=lambda x: x[0], reverse=True)

    top = scored[:TOPK_EVIDENCE]
    out: List[Evidence] = []

    # include neighbor window in reading order (page-local)
    ro_sorted = sorted(cand, key=lambda s: s.reading_order)
    idx_by_id = {s.span_id: i for i, s in enumerate(ro_sorted)}

    seen = set()
    for score, s in top:
        i = idx_by_id.get(s.span_id, None)
        for j in range(max(0, i-NEIGHBOR_WINDOW), min(len(ro_sorted), i+NEIGHBOR_WINDOW+1)):
            sj = ro_sorted[j]
            if sj.span_id in seen: 
                continue
            seen.add(sj.span_id)
            out.append(Evidence(span_id=sj.span_id, page=sj.page, bbox_norm=sj.bbox_norm, text=sj.text, score=float(score)))
    # sort evidence by page reading order for nicer display
    out.sort(key=lambda e: idx_by_id.get(e.span_id, 10**9))
    return out


def bbox_iou(a: BBox, b: BBox) -> float:
    """Compute intersection-over-union of two bboxes (x0, y0, x1, y1)."""
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def resolve_selections(doc_id: str, selections: list[dict]) -> list[dict]:
    """Find spans overlapping with bbox selections, ranked by IoU."""
    p = paths(doc_id)
    spans = read_spans_jsonl(p["spans"])

    seen: set[str] = set()
    results: list[dict] = []

    for sel in selections:
        page = sel["page"]
        sel_bbox = tuple(sel["bbox_norm"])
        page_spans = [s for s in spans if s.page == page]

        scored = []
        for s in page_spans:
            iou = bbox_iou(sel_bbox, s.bbox_norm)
            if iou > 0 and s.span_id not in seen:
                scored.append((iou, s))

        scored.sort(key=lambda x: x[0], reverse=True)
        for iou, s in scored:
            seen.add(s.span_id)
            results.append({
                "span_id": s.span_id,
                "page": s.page,
                "text": s.text,
                "bbox_norm": s.bbox_norm,
                "iou": round(iou, 4),
            })

    return results

