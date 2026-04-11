"""Parser dispatch — picks an implementation based on METIS_INGEST_PARSER."""
from __future__ import annotations

from ...settings import INGEST_PARSER
from .base import Parser


def get_parser(name: str | None = None) -> Parser:
    """Return a Parser implementation by name. Defaults to METIS_INGEST_PARSER."""
    requested = (name or INGEST_PARSER).lower()
    if requested == "docling":
        from .docling_parser import DoclingParser
        return DoclingParser()
    if requested in {"pymupdf4llm", "pymupdf", "legacy"}:
        from .pymupdf_parser import PyMuPDFParser
        return PyMuPDFParser()
    raise ValueError(
        f"Unknown parser: {requested!r}. "
        f"Valid values: 'docling', 'pymupdf4llm'."
    )
