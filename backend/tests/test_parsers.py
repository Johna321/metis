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


def test_docling_parser_no_duplicate_paragraph_ids_in_heading_lists(real_paper_bytes):
    """Regression test: every heading's paragraph_ids list has unique entries,
    and no paragraph is referenced by more than one heading."""
    parser = get_parser("docling")
    tree = parser.parse(real_paper_bytes, doc_id="sha256:late-chunking")

    # No duplicates within any heading's list
    for sec_id, h in tree.headings.items():
        assert len(h.paragraph_ids) == len(set(h.paragraph_ids)), \
            f"Heading {sec_id} has duplicate paragraph_ids: {h.paragraph_ids}"

    # No paragraph reachable from more than one heading
    seen: set[str] = set()
    for h in tree.headings.values():
        for pid in h.paragraph_ids:
            assert pid not in seen, f"Paragraph {pid} reachable from multiple headings"
            seen.add(pid)

    # Every paragraph in tree.paragraphs is reachable from exactly one heading
    assert seen == set(tree.paragraphs.keys()), \
        f"Paragraphs not matching: in_paragraphs={len(tree.paragraphs)}, in_headings={len(seen)}"


def test_docling_parser_title_not_in_heading_stack(real_paper_bytes):
    """Docling 'title' items should become tree.title, not a HeadingNode."""
    parser = get_parser("docling")
    tree = parser.parse(real_paper_bytes, doc_id="sha256:late-chunking")
    # The paper title should populate tree.title (not just 'input' from the stream name)
    assert tree.title != "input"
    assert tree.title != "Untitled"
    # For the Late Chunking paper, the true title starts with "LATE CHUNKING:"
    assert "LATE CHUNKING" in tree.title.upper(), \
        f"Expected paper title, got: {tree.title!r}"
    # None of the HeadingNodes should duplicate the paper title verbatim —
    # the title must live only on tree.title, not on a level-1 heading.
    for h in tree.headings.values():
        if h.sec_id == "root":
            continue
        assert h.title.strip() != tree.title.strip(), \
            f"Heading {h.sec_id} duplicates the paper title: {h.title!r}"


def test_docling_parser_abstract_gets_reserved_sec_id(real_paper_bytes):
    """Well-known unnumbered sections should use reserved sec_ids, not autonumber."""
    parser = get_parser("docling")
    tree = parser.parse(real_paper_bytes, doc_id="sha256:late-chunking")

    # Find a heading whose title starts with "Abstract" — it must have sec_id "abstract"
    abstract_headings = [
        h for h in tree.headings.values()
        if h.title.lower().startswith("abstract")
    ]
    if abstract_headings:
        assert any(h.sec_id == "abstract" for h in abstract_headings), (
            f"Abstract heading has sec_id {[h.sec_id for h in abstract_headings]}, "
            f"expected 'abstract'"
        )

    # Section "1" should be Introduction (real numbered section), not Abstract
    sec1 = tree.headings.get("1")
    if sec1 is not None:
        assert "abstract" not in sec1.title.lower(), (
            f"sec_id '1' should be the real Introduction, got title: {sec1.title!r}"
        )
        # Should start with "1 " (dotted number prefix)
        assert sec1.title.strip().startswith("1"), (
            f"sec_id '1' title should start with '1', got: {sec1.title!r}"
        )

    # References should use "refs" if present
    refs_headings = [
        h for h in tree.headings.values()
        if h.title.lower().startswith(("references", "bibliography"))
    ]
    if refs_headings:
        assert any(h.sec_id == "refs" for h in refs_headings), (
            f"References has sec_id {[h.sec_id for h in refs_headings]}, expected 'refs'"
        )


def test_pymupdf_parser_returns_doctree(pdf_bytes):
    parser = get_parser("pymupdf4llm")
    tree = parser.parse(pdf_bytes, doc_id="sha256:synthetic")
    assert tree.doc_id == "sha256:synthetic"
    # Synthetic 1-page PDF has 4 sentences — we should get 4 paragraphs
    assert len(tree.paragraphs) >= 1
    # Root must exist
    assert "root" in tree.headings
    # parse_meta records it
    assert tree.parse_meta.get("parser") == "pymupdf4llm"
