from __future__ import annotations
from typing import List
from rapidfuzz import fuzz
from .schema import Evidence, Span
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

