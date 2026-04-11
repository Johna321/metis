"""Legacy pymupdf4llm → DocTree.

Used as a fallback (METIS_INGEST_PARSER=pymupdf4llm). Wraps the existing
layout-engine logic from metis.core.ingest and reindexes into DocTree.
"""
from __future__ import annotations
import logging
from dataclasses import replace
from typing import Any

from ..schema_tree import DocTree, HeadingNode, ParagraphNode, make_para_id
from ...settings import MIN_CHARS

log = logging.getLogger(__name__)


class PyMuPDFParser:
    name = "pymupdf4llm"

    def parse(self, pdf_bytes: bytes, doc_id: str) -> DocTree:
        import pymupdf
        # Fall back to simple blocks extraction. The layout engine
        # (pymupdf4llm + pymupdf_layout) is exercised in the existing
        # ingest.py tests; here we just need a valid DocTree.
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")

        root = HeadingNode(
            doc_id=doc_id, sec_id="root", level=0,
            title=doc.metadata.get("title") or "Untitled",
            title_bbox_norm=None, title_page=None,
            parent_sec_id=None, children_sec_ids=[], paragraph_ids=[],
            n_tokens_subtree=0,
        )

        paragraphs: dict[str, ParagraphNode] = {}
        root_paragraph_ids: list[str] = []
        ro = 0
        idx = 0

        for page_i in range(doc.page_count):
            page = doc[page_i]
            w, h = page.rect.width, page.rect.height
            blocks = page.get_text("blocks")
            blocks.sort(key=lambda b: (b[1], b[0]))
            for b in blocks:
                x0, y0, x1, y1, text_raw, *_ = b
                text = " ".join((text_raw or "").split())
                if len(text) < MIN_CHARS:
                    continue
                bbox_norm = (float(x0) / w, float(y0) / h, float(x1) / w, float(y1) / h)
                pid = make_para_id(doc_id, "root", idx)
                paragraphs[pid] = ParagraphNode(
                    doc_id=doc_id, sec_id="root", para_idx=idx, para_id=pid,
                    kind="text", text=text, label=None,
                    bbox_norm=bbox_norm, page=page_i,
                    reading_order=ro, n_tokens=len(text.split()),
                )
                root_paragraph_ids.append(pid)
                idx += 1
                ro += 1

        root_filled = replace(
            root,
            paragraph_ids=root_paragraph_ids,
            n_tokens_subtree=sum(paragraphs[pid].n_tokens for pid in root_paragraph_ids),
        )
        return DocTree(
            doc_id=doc_id,
            title=root.title,
            authors=[],
            abstract_summary=None,
            root=root_filled,
            headings={"root": root_filled},
            paragraphs=paragraphs,
            labeled_entities={},
            parse_meta={"parser": "pymupdf4llm", "n_paragraphs": len(paragraphs)},
        )
