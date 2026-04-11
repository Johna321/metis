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
        raise NotImplementedError("Implemented in Task 5")
