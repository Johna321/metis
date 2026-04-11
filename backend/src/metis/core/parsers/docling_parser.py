"""Docling → DocTree.

Uses Docling's typed DoclingDocument (RT-DETR layout + TableFormer) and
reindexes into our coordinate-addressed DocTree schema.

Stages:
    1. Convert PDF bytes to DoclingDocument via DocumentConverter.
    2. Walk the document in reading order, emit HeadingNode and ParagraphNode.
    3. Level-repair pass using dotted numbering in heading titles.
    4. Assemble DocTree.
"""
from __future__ import annotations
import logging
from dataclasses import replace
from typing import Any

from ..schema_tree import DocTree, HeadingNode, ParagraphNode, make_para_id

log = logging.getLogger(__name__)


class DoclingParser:
    """Parser implementation backed by IBM Docling."""

    name = "docling"

    def __init__(self) -> None:
        from docling.document_converter import DocumentConverter
        self._converter = DocumentConverter()

    def parse(self, pdf_bytes: bytes, doc_id: str) -> DocTree:
        docling_doc = self._convert(pdf_bytes)
        return self._docling_to_doctree(docling_doc, doc_id)

    def _convert(self, pdf_bytes: bytes):
        """Run Docling over pdf_bytes. Returns a DoclingDocument."""
        import io
        from docling.datamodel.base_models import DocumentStream
        # DocumentStream wraps a BytesIO so we don't have to write a temp file
        stream = DocumentStream(name="input.pdf", stream=io.BytesIO(pdf_bytes))
        result = self._converter.convert(stream)
        return result.document

    def _docling_to_doctree(self, docling_doc: Any, doc_id: str) -> DocTree:
        # Docling provides `iterate_items()` which yields items in reading order.
        # Items have a `label` (e.g., "section_header", "text", "formula", "table",
        # "picture", "caption", "page_header", "page_footer") and `prov` with bbox
        # + page. Section headers form the hierarchy; we build a stack.
        from docling_core.types.doc import DocItemLabel  # type: ignore  # noqa: F401
        import re

        # Known labels to drop entirely.
        _DROP_LABELS = {"page_header", "page_footer"}

        # Map Docling label → our ParagraphNode.kind
        _KIND_MAP = {
            "text": "text",
            "list_item": "list-item",
            "formula": "formula",
            "table": "table",
            "picture": "figure",
            "caption": "caption",
            "footnote": "text",
        }

        # Build the root
        title_text = docling_doc.name or "Untitled"
        root = HeadingNode(
            doc_id=doc_id, sec_id="root", level=0, title=title_text,
            title_bbox_norm=None, title_page=None,
            parent_sec_id=None, children_sec_ids=[], paragraph_ids=[],
            n_tokens_subtree=0,
        )
        headings: dict[str, HeadingNode] = {"root": root}
        # Mutable bookkeeping lists (we replace() headings at the end)
        h_children: dict[str, list[str]] = {"root": []}
        h_paragraphs: dict[str, list[str]] = {"root": []}
        paragraphs: dict[str, ParagraphNode] = {}

        stack: list[str] = ["root"]        # sec_id stack
        stack_levels: list[int] = [0]
        ro = 0                              # global reading order
        section_counters: dict[int, int] = {1: 0}  # auto-numbered fallback per level
        para_counters: dict[str, int] = {"root": 0}  # next para_idx under each sec_id
        level_repairs = 0

        def _bbox_norm_from_prov(prov, page_w: float, page_h: float) -> tuple[float, float, float, float]:
            # Normalize bbox to [0, 1] with top-left origin.
            # Docling's BoundingBox has `to_top_left_origin(page_height)` which flips
            # BOTTOMLEFT → TOPLEFT correctly; TOPLEFT inputs pass through unchanged.
            b = prov.bbox.to_top_left_origin(page_height=page_h)
            x0 = float(b.l) / page_w
            y0 = float(b.t) / page_h
            x1 = float(b.r) / page_w
            y1 = float(b.b) / page_h
            return (x0, y0, x1, y1)

        def _push_heading(level: int, title: str, bbox, page):
            nonlocal level_repairs
            # Level repair via dotted numbering: if title matches "\d+(\.\d+)*\s+..."
            m = re.match(r"^(\d+(?:\.\d+)*)[\s.:]+(.+)$", title.strip())
            if m:
                number = m.group(1)
                repaired_level = number.count(".") + 1
                if repaired_level != level:
                    level_repairs += 1
                level = repaired_level
                sec_id = number
            else:
                # Fall back to autonumbering at the detected level
                section_counters[level] = section_counters.get(level, 0) + 1
                # Build synthetic id from stack
                parent = stack[-1] if stack else "root"
                sec_id = f"{parent}.{section_counters[level]}" if parent != "root" else str(section_counters[level])
                # Reset deeper counters
                for lv in list(section_counters.keys()):
                    if lv > level:
                        section_counters[lv] = 0

            # Pop stack until parent level is shallower than `level`
            while stack_levels and stack_levels[-1] >= level:
                stack.pop()
                stack_levels.pop()
            parent = stack[-1] if stack else "root"

            h = HeadingNode(
                doc_id=doc_id, sec_id=sec_id, level=level, title=title.strip(),
                title_bbox_norm=bbox, title_page=page,
                parent_sec_id=parent, children_sec_ids=[], paragraph_ids=[],
                n_tokens_subtree=0,
            )
            headings[sec_id] = h
            h_children.setdefault(sec_id, [])
            h_paragraphs.setdefault(sec_id, [])
            h_children.setdefault(parent, []).append(sec_id)
            para_counters[sec_id] = 0

            stack.append(sec_id)
            stack_levels.append(level)

        def _add_paragraph(kind: str, text: str, bbox, page):
            nonlocal ro
            parent = stack[-1] if stack else "root"
            idx = para_counters.get(parent, 0)
            para_counters[parent] = idx + 1
            pid = make_para_id(doc_id, parent, idx)
            p = ParagraphNode(
                doc_id=doc_id, sec_id=parent, para_idx=idx, para_id=pid,
                kind=kind, text=text, label=None,
                bbox_norm=bbox, page=page,
                reading_order=ro, n_tokens=len(text.split()),
            )
            paragraphs[pid] = p
            h_paragraphs.setdefault(parent, []).append(pid)
            ro += 1

        # Walk the document. Docling's iterate_items yields (item, level) pairs.
        for item, _level in docling_doc.iterate_items():
            label = getattr(item.label, "value", str(item.label)) if hasattr(item, "label") else "text"
            if label in _DROP_LABELS:
                continue
            # Extract text
            text = (getattr(item, "text", "") or "").strip()
            # Get bbox + page from prov (if present)
            prov = item.prov[0] if getattr(item, "prov", None) else None
            # Docling pages are keyed 1-indexed; we store 0-indexed on ParagraphNode.
            page_no_1idx = prov.page_no if prov and hasattr(prov, "page_no") else 1
            page = page_no_1idx - 1
            # Look up page dimensions
            if prov and docling_doc.pages and page_no_1idx in docling_doc.pages:
                page_obj = docling_doc.pages[page_no_1idx]
                page_w = float(page_obj.size.width)
                page_h = float(page_obj.size.height)
                bbox = _bbox_norm_from_prov(prov, page_w, page_h)
            else:
                bbox = (0.0, 0.0, 1.0, 1.0)

            if label in ("section_header", "title"):
                if not text:
                    continue
                # Docling doesn't always give explicit levels. Default to 1.
                level = 1
                _push_heading(level, text, bbox, page)
            elif label in _KIND_MAP:
                if label != "picture" and not text:
                    continue
                _add_paragraph(_KIND_MAP[label], text or "[[PICTURE]]", bbox, page)
            else:
                if text:
                    _add_paragraph("text", text, bbox, page)

        # Re-materialize headings with filled children/paragraphs lists.
        for sec_id, h in list(headings.items()):
            headings[sec_id] = replace(
                h,
                children_sec_ids=list(h_children.get(sec_id, [])),
                paragraph_ids=list(h_paragraphs.get(sec_id, [])),
                n_tokens_subtree=sum(paragraphs[pid].n_tokens for pid in h_paragraphs.get(sec_id, [])),
            )

        return DocTree(
            doc_id=doc_id,
            title=title_text,
            authors=[],  # Docling does not reliably surface authors; populate later if available
            abstract_summary=None,
            root=headings["root"],
            headings=headings,
            paragraphs=paragraphs,
            labeled_entities={},  # populated in Task 7
            parse_meta={
                "parser": "docling",
                "level_repairs": level_repairs,
                "n_headings": len(headings),
                "n_paragraphs": len(paragraphs),
            },
        )
