from __future__ import annotations
import logging
import pymupdf
from collections import Counter
from typing import Dict, List, Tuple
from .schema import Span
from .store import doc_id_from_bytes, paths, write_json, write_spans_jsonl
from ..settings import MIN_CHARS

log = logging.getLogger(__name__)

def _norm_bbox(b: Tuple[float,float,float,float], w: float, h: float):
    x0,y0,x1,y1 = b
    return (x0/w, y0/h, x1/w, y1/h)

# ---------------------------------------------------------------------------
# blocks-based ingestion 
# ---------------------------------------------------------------------------

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
                source="pymupdf_blocks",
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

# ---------------------------------------------------------------------------
# layout-based ingestion via pymupdf4llm
# ---------------------------------------------------------------------------

def ensure_pymupdf4llm():
    """Import pymupdf4llm with correct init order. Raises RuntimeError if missing."""
    # pymupdf.layout (aka pymupdf_layout) improves layout analysis if installed;
    # it must be imported before pymupdf4llm so the latter can detect it.
    # NOTE: pymupdf.layout is optional ... pymupdf4llm works without it.
    try:
        import importlib
        importlib.import_module("pymupdf.layout")
    except ImportError:
        pass
    try:
        import pymupdf4llm
        return pymupdf4llm
    except ImportError as exc:
        raise RuntimeError(
            "pymupdf4llm is required for the layout engine. "
            "Install it with:  pip install pymupdf4llm"
        ) from exc


def _find_text_pos(page_text: str, needle: str) -> Tuple[int, int] | None:
    """Try to find normalized needle text in page markdown. Returns (start, end) or None.

    Best-effort heuristic: tries full text first, falls back to prefix matching.
    May return None or imprecise offsets for repeated/short text.
    """
    if not needle:
        return None
    # Try exact match first
    idx = page_text.find(needle)
    if idx != -1:
        return (idx, idx + len(needle))
    # Fall back to prefix-based search for texts that differ due to markdown formatting
    prefix = needle[:60].strip()
    if not prefix:
        return None
    idx = page_text.find(prefix)
    if idx == -1:
        return None
    # Search for the suffix after the prefix match to estimate the end
    if len(needle) > 60:
        suffix = needle[-40:].strip()
        end_idx = page_text.find(suffix, idx + len(prefix))
        if end_idx != -1:
            return (idx, end_idx + len(suffix))
    return (idx, idx + len(prefix))


def _rect_to_tuple(r) -> Tuple[float, float, float, float]:
    """Convert a pymupdf Rect (or tuple/list) to a plain 4-tuple of floats."""
    if hasattr(r, "x0"):  # pymupdf.Rect
        return (float(r.x0), float(r.y0), float(r.x1), float(r.y1))
    return tuple(float(v) for v in r[:4])


def ingest_pdf_bytes_layout(
    pdf_bytes: bytes,
    *,
    extract_words: bool = False,
    write_images: bool = False,
    dpi: int = 200,
) -> dict:
    """Ingest a PDF using pymupdf4llm for layout-aware spans.

    With pymupdf_layout installed, chunks contain `page_boxes` with classified
    regions (text, title, picture, section-header, caption, etc.) and char
    offsets into the page markdown. Without it, falls back to pymupdf blocks
    + separate tables/images/graphics lists.
    """
    pymupdf4llm = ensure_pymupdf4llm()

    doc_id = doc_id_from_bytes(pdf_bytes)
    p = paths(doc_id)
    p["pdf"].write_bytes(pdf_bytes)

    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")

    # Prepare image output directory if requested
    image_path = None
    if write_images:
        image_path = p["assets"] / "images"
        image_path.mkdir(parents=True, exist_ok=True)

    # Run pymupdf4llm extraction
    md_kwargs: dict = dict(
        page_chunks=True,
        force_text=True,
    )
    if extract_words:
        md_kwargs["extract_words"] = True
    if write_images and image_path is not None:
        md_kwargs["write_images"] = True
        md_kwargs["image_path"] = str(image_path)
        md_kwargs["dpi"] = dpi

    chunks = pymupdf4llm.to_markdown(doc, **md_kwargs)

    spans: List[Span] = []
    page_md: Dict[str, str] = {}    # str(page_i) -> markdown text (str keys for JSON)
    words_by_page: Dict[str, list] = {}
    ro = 0
    total_counter: Counter = Counter()

    for page_i, chunk in enumerate(chunks):
        page = doc[page_i]
        w, h = page.rect.width, page.rect.height

        page_text: str = chunk.get("text", "")
        page_md[str(page_i)] = page_text

        if extract_words and "words" in chunk:
            # words are tuples: (x0, y0, x1, y1, word, block_no, line_no, word_no)
            # store as-is for debug rendering
            words_by_page[str(page_i)] = chunk["words"]

        page_counter: Counter = Counter()
        page_boxes = chunk.get("page_boxes")

        if page_boxes is not None:
            # --- pymupdf_layout path: iterate classified page_boxes ---
            for li, box in enumerate(page_boxes):
                kind = box.get("class") or box.get("type") or "unknown"
                bbox_raw = box.get("bbox")
                if bbox_raw is None:
                    log.warning("page %d box %d (%s): no bbox, skipping", page_i, li, kind)
                    continue
                bbox_pdf = _rect_to_tuple(bbox_raw)

                pos_raw = box.get("pos")
                pos = tuple(int(v) for v in pos_raw) if pos_raw is not None else None

                # Extract span text from page markdown via pos
                if pos is not None and kind != "picture":
                    text = page_text[pos[0]:pos[1]]
                    text = " ".join(text.split())
                elif kind == "picture":
                    text = "[[PICTURE]]"
                else:
                    text = ""

                # MIN_CHARS filter only for text-like kinds
                if kind in ("text", "table") and len(text) < MIN_CHARS:
                    continue

                page_counter[kind] += 1
                span = Span(
                    span_id=f"p{page_i:03d}_L{li:04d}",
                    doc_id=doc_id,
                    page=page_i,
                    bbox_pdf=bbox_pdf,
                    bbox_norm=_norm_bbox(bbox_pdf, w, h),
                    text=text,
                    reading_order=ro,
                    is_header=(kind == "page-header"),
                    is_footer=(kind == "page-footer"),
                    kind=kind,
                    pos=pos,
                    source="pymupdf4llm_page_boxes",
                )
                spans.append(span)
                ro += 1
        else:
            # --- Fallback path (no pymupdf_layout): blocks + tables/images ---
            li = 0
            # Text blocks from pymupdf
            blocks = page.get_text("blocks")
            blocks.sort(key=lambda b: (b[1], b[0]))
            for bi, b in enumerate(blocks):
                x0, y0, x1, y1, text_raw, *rest = b
                block_type = rest[1] if len(rest) > 1 else 0
                if block_type != 0:
                    continue
                t = " ".join(text_raw.split())
                if len(t) < MIN_CHARS:
                    continue
                bbox_pdf = (float(x0), float(y0), float(x1), float(y1))
                pos = _find_text_pos(page_text, t)
                page_counter["text"] += 1
                spans.append(Span(
                    span_id=f"p{page_i:03d}_L{li:04d}",
                    doc_id=doc_id, page=page_i,
                    bbox_pdf=bbox_pdf, bbox_norm=_norm_bbox(bbox_pdf, w, h),
                    text=t, reading_order=ro,
                    kind="text", pos=pos, source="pymupdf4llm_layout",
                ))
                ro += 1; li += 1

            # Tables
            for tbl in (chunk.get("tables") or []):
                bbox_raw = tbl.get("bbox")
                if bbox_raw is None:
                    continue
                bbox_pdf = _rect_to_tuple(bbox_raw)
                rect = pymupdf.Rect(bbox_pdf)
                tbl_text = " ".join(page.get_text("text", clip=rect).split())
                page_counter["table"] += 1
                spans.append(Span(
                    span_id=f"p{page_i:03d}_L{li:04d}",
                    doc_id=doc_id, page=page_i,
                    bbox_pdf=bbox_pdf, bbox_norm=_norm_bbox(bbox_pdf, w, h),
                    text=tbl_text if len(tbl_text) >= MIN_CHARS else f"[[TABLE {tbl.get('rows',0)}x{tbl.get('columns',0)}]]",
                    reading_order=ro, kind="table", source="pymupdf4llm_layout",
                ))
                ro += 1; li += 1

            # Images
            for img in (chunk.get("images") or []):
                bbox_raw = img.get("bbox")
                if bbox_raw is None:
                    continue
                bbox_pdf = _rect_to_tuple(bbox_raw)
                page_counter["picture"] += 1
                spans.append(Span(
                    span_id=f"p{page_i:03d}_L{li:04d}",
                    doc_id=doc_id, page=page_i,
                    bbox_pdf=bbox_pdf, bbox_norm=_norm_bbox(bbox_pdf, w, h),
                    text="[[PICTURE]]", reading_order=ro,
                    kind="picture", source="pymupdf4llm_layout",
                ))
                ro += 1; li += 1

            # Graphics
            for gfx in (chunk.get("graphics") or []):
                bbox_raw = gfx.get("bbox")
                if bbox_raw is None:
                    continue
                bbox_pdf = _rect_to_tuple(bbox_raw)
                page_counter["graphic"] += 1
                spans.append(Span(
                    span_id=f"p{page_i:03d}_L{li:04d}",
                    doc_id=doc_id, page=page_i,
                    bbox_pdf=bbox_pdf, bbox_norm=_norm_bbox(bbox_pdf, w, h),
                    text="[[GRAPHIC]]", reading_order=ro,
                    kind="graphic", source="pymupdf4llm_layout",
                ))
                ro += 1; li += 1

        total_counter += page_counter
        log.info("page %d: %s", page_i, dict(page_counter))

    log.info("total spans: %d, region counts: %s", len(spans), dict(total_counter))

    meta = {
        "doc_id": doc_id,
        "n_pages": doc.page_count,
        "n_spans": len(spans),
        "ingest": {
            "engine": "pymupdf4llm",
            "min_chars": MIN_CHARS,
            "extract_words": extract_words,
            "write_images": write_images,
            "dpi": dpi,
        },
    }

    write_json(p["doc"], meta)
    write_spans_jsonl(p["spans"], spans)
    write_json(p["page_md"], page_md)

    # Store words if extracted (sidecar file next to page_md)
    if words_by_page:
        words_path = p["page_md"].with_suffix(".words.json")
        write_json(words_path, words_by_page)

    return meta
