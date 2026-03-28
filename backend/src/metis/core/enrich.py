"""Multimodal span enrichment — render visual regions and extract structured content."""
from __future__ import annotations
import dataclasses
import logging
from typing import TYPE_CHECKING

import pymupdf

from .schema import Span
from .store import paths

if TYPE_CHECKING:
    from PIL import Image as PILImage

log = logging.getLogger(__name__)

ENRICHABLE_KINDS = {"formula", "table"}

# ---------------------------------------------------------------------------
# Pix2text lazy singleton
# ---------------------------------------------------------------------------

_p2t_instance = None
_P2T_NOT_INSTALLED = object()  # sentinel


def _get_p2t():
    """Return the Pix2Text instance, or None if not installed."""
    global _p2t_instance
    if _p2t_instance is None:
        try:
            from pix2text import Pix2Text
            _p2t_instance = Pix2Text.from_config(
                enable_formula=True,
                enable_table=True,
                device="cpu",
            )
        except ImportError:
            log.info("pix2text not installed — multimodal enrichment disabled")
            _p2t_instance = _P2T_NOT_INSTALLED
    return _p2t_instance if _p2t_instance is not _P2T_NOT_INSTALLED else None


# ---------------------------------------------------------------------------
# Bbox rendering
# ---------------------------------------------------------------------------

def _render_bbox(
    doc: pymupdf.Document,
    *,
    page: int,
    bbox_pdf: tuple[float, float, float, float],
    dpi: int = 200,
) -> "PILImage.Image":
    """Render a bounding box region from a PDF page as a PIL image."""
    from PIL import Image

    pg = doc[page]
    clip = pymupdf.Rect(bbox_pdf)
    pixmap = pg.get_pixmap(clip=clip, dpi=dpi)
    return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)


# ---------------------------------------------------------------------------
# Asset saving
# ---------------------------------------------------------------------------

def _save_asset(image: "PILImage.Image", doc_id: str, span_id: str) -> str:
    """Save rendered bbox image to _assets/images/. Returns relative path from DATA_DIR."""
    p = paths(doc_id)
    images_dir = p["assets"] / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{span_id}.png"
    full_path = images_dir / filename
    image.save(full_path)

    # Return path relative to DATA_DIR (p["assets"].parent is DATA_DIR)
    rel = full_path.relative_to(p["assets"].parent)
    return str(rel)


# ---------------------------------------------------------------------------
# Per-kind enrichment
# ---------------------------------------------------------------------------

def _enrich_formula(p2t, image: "PILImage.Image") -> str:
    """Run pix2text MFR on a formula image, return LaTeX wrapped in $$."""
    latex = p2t.recognize_formula(image, return_text=True)
    if isinstance(latex, str):
        latex = latex.strip()
    else:
        latex = str(latex).strip()
    return f"$${latex}$$"


def _enrich_table(p2t, image: "PILImage.Image") -> str:
    """Run pix2text table recognition on a table image, return markdown."""
    if p2t.table_ocr is None:
        raise RuntimeError("table_ocr not initialized — use Pix2Text.from_config(enable_table=True)")
    result = p2t.table_ocr.recognize(image, out_markdown=True)
    md_list = result.get("markdown", [])
    if md_list:
        return md_list[0].strip()
    return ""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def enrich_visual_spans(
    spans: list[Span],
    pdf_bytes: bytes,
) -> list[Span]:
    """Process visual spans through pix2text extractors.

    Returns a new list with enriched spans replacing originals.
    If pix2text is not installed, returns spans unchanged.
    """
    p2t = _get_p2t()
    if p2t is None:
        return spans

    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    enriched: list[Span] = []

    for span in spans:
        if span.kind not in ENRICHABLE_KINDS:
            enriched.append(span)
            continue

        # Render bbox as image
        try:
            image = _render_bbox(
                doc,
                page=span.page,
                bbox_pdf=span.bbox_pdf,
            )
        except Exception:
            log.warning("Failed to render bbox for %s", span.span_id, exc_info=True)
            enriched.append(span)
            continue

        # Save asset image
        asset_path = None
        try:
            asset_path = _save_asset(image, span.doc_id, span.span_id)
        except Exception:
            log.warning("Failed to save asset for %s", span.span_id, exc_info=True)

        # Dispatch to extractor by kind
        try:
            if span.kind == "formula":
                new_text = _enrich_formula(p2t, image)
                content_source = "pix2text_mfr"
            elif span.kind == "table":
                new_text = _enrich_table(p2t, image)
                content_source = "pix2text_table"
            else:
                enriched.append(span)
                continue

            enriched.append(dataclasses.replace(
                span,
                text=new_text,
                asset_path=asset_path,
                content_source=content_source,
                original_text=span.text,
            ))
            log.info("Enriched %s (%s) via %s", span.span_id, span.kind, content_source)

        except Exception:
            log.warning("Enrichment failed for %s (%s)", span.span_id, span.kind, exc_info=True)
            if asset_path:
                enriched.append(dataclasses.replace(span, asset_path=asset_path))
            else:
                enriched.append(span)

    doc.close()
    return enriched
