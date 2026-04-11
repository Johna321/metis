"""Integration tests for parsers. Uses real PDFs from metis-reading/ when available,
falls back to the synthetic pdf_bytes fixture."""
from __future__ import annotations
from pathlib import Path
import pytest

from metis.core.parsers import get_parser
from metis.core.schema_tree import DocTree, HeadingNode, ParagraphNode


REAL_PAPER = Path("../../metis-reading/Late Chunking_ Contextual Chunk Embeddings Using Long-Context Embedding Models.pdf")


@pytest.fixture
def real_paper_bytes() -> bytes:
    if not REAL_PAPER.exists():
        pytest.skip(f"Real paper not found at {REAL_PAPER}")
    return REAL_PAPER.read_bytes()


def test_docling_parser_returns_doctree(real_paper_bytes):
    parser = get_parser("docling")
    tree = parser.parse(real_paper_bytes, doc_id="sha256:late-chunking")
    assert isinstance(tree, DocTree)
    assert tree.doc_id == "sha256:late-chunking"
    # A real paper must have at least some sections and paragraphs.
    assert len(tree.headings) > 1, "Expected more than just the root heading"
    assert len(tree.paragraphs) > 5, "Expected multiple paragraphs"
    # Every paragraph must be reachable from a heading.
    reached = {pid for h in tree.headings.values() for pid in h.paragraph_ids}
    assert set(tree.paragraphs.keys()) <= reached, "Some paragraphs are orphaned"
    # Parse meta records which parser was used.
    assert tree.parse_meta.get("parser") == "docling"


def test_docling_parser_emits_numbered_sections(real_paper_bytes):
    parser = get_parser("docling")
    tree = parser.parse(real_paper_bytes, doc_id="sha256:late-chunking")
    # Late Chunking paper has Introduction, Method, Evaluation sections.
    # At least one top-level numbered heading should exist.
    numbered = [h for h in tree.headings.values() if h.level == 1 and h.sec_id.split(".")[0].isdigit()]
    assert len(numbered) >= 1, f"Expected numbered sections; got: {[h.sec_id for h in tree.headings.values()]}"
