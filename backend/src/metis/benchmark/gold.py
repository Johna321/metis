"""Gold annotation schema helpers for PDF ingestion benchmarks."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import orjson

from ..core.store import paths, read_spans_jsonl


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def bootstrap_from_spans(doc_id: str) -> dict:
    """Create annotation state from ingested spans, all marked pending."""
    p = paths(doc_id)
    spans = read_spans_jsonl(p["spans"])

    pages: dict[str, list[dict]] = {}
    for s in spans:
        page_key = str(s.page)
        ann = {
            "ann_id": f"a_{s.span_id}",
            "bbox_norm": list(s.bbox_norm),
            "bbox_pdf": list(s.bbox_pdf),
            "kind": s.kind or "text",
            "reading_order": s.reading_order,
            "text": s.text,
            "status": "pending",
            "provenance": "engine",
            "group_id": None,
            "source_span_id": s.span_id,
        }
        pages.setdefault(page_key, []).append(ann)

    return {
        "doc_id": doc_id,
        "version": 1,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "pages": pages,
    }


def load_annotation_state(doc_id: str) -> dict | None:
    """Load annotation state from disk, or None if not found."""
    p = paths(doc_id)
    path = p["annotation"]
    if not path.exists():
        return None
    return orjson.loads(path.read_bytes())


def save_annotation_state(doc_id: str, state: dict) -> None:
    """Persist annotation state to disk."""
    state["updated_at"] = _now_iso()
    p = paths(doc_id)
    p["annotation"].write_bytes(orjson.dumps(state, option=orjson.OPT_INDENT_2))


def export_to_gold(state: dict) -> Path:
    """Export accepted annotations to a .gold.json file.

    Returns the path to the written file.
    """
    doc_id = state["doc_id"]
    gold_pages: dict[str, list[dict]] = {}

    for page_key, anns in state["pages"].items():
        accepted = []
        for a in anns:
            if a["status"] != "accepted":
                continue
            entry: dict = {
                "bbox_norm": a["bbox_norm"],
                "kind": a["kind"],
                "reading_order": a["reading_order"],
            }
            if a.get("group_id"):
                entry["group_id"] = a["group_id"]
            accepted.append(entry)
        if accepted:
            # Sort by reading_order for consistency
            accepted.sort(key=lambda x: x["reading_order"])
            gold_pages[page_key] = accepted

    gold = {
        "doc_id": doc_id,
        "pages": gold_pages,
    }

    p = paths(doc_id)
    out_path = p["gold"]
    out_path.write_bytes(orjson.dumps(gold, option=orjson.OPT_INDENT_2))
    return out_path
