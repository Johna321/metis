"""Tests for DocTree schema types."""
from __future__ import annotations
import dataclasses
import pytest

from metis.core.schema_tree import HeadingNode, ParagraphNode, DocTree


def test_paragraph_node_canonical_id():
    p = ParagraphNode(
        doc_id="sha256:abc",
        sec_id="3.2",
        para_idx=4,
        para_id="sha256:abc::3.2::p4",
        kind="text",
        text="We optimize the policy...",
        label=None,
        bbox_norm=(0.1, 0.2, 0.9, 0.3),
        page=5,
        reading_order=42,
        n_tokens=24,
    )
    assert p.para_id == "sha256:abc::3.2::p4"
    assert p.kind == "text"
    # Frozen — mutation raises
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.text = "other"  # type: ignore[misc]


def test_paragraph_node_enrichment_fields_default_none():
    p = ParagraphNode(
        doc_id="sha256:abc", sec_id="3.2", para_idx=0,
        para_id="sha256:abc::3.2::p0", kind="formula",
        text="$$x^2$$", label="Equation 1",
        bbox_norm=(0, 0, 1, 1), page=0, reading_order=0, n_tokens=3,
    )
    assert p.asset_path is None
    assert p.content_source is None
    assert p.original_text is None


def test_heading_node_children_and_paragraphs():
    h = HeadingNode(
        doc_id="sha256:abc",
        sec_id="3",
        level=1,
        title="Model Architecture",
        title_bbox_norm=(0.1, 0.05, 0.5, 0.08),
        title_page=3,
        parent_sec_id="root",
        children_sec_ids=["3.1", "3.2"],
        paragraph_ids=["sha256:abc::3::p0"],
        n_tokens_subtree=512,
    )
    assert h.children_sec_ids == ["3.1", "3.2"]
    assert h.level == 1


def test_make_para_id_format():
    from metis.core.schema_tree import make_para_id
    assert make_para_id("sha256:abc", "3.2", 4) == "sha256:abc::3.2::p4"
    assert make_para_id("sha256:xyz", "root", 0) == "sha256:xyz::root::p0"


def test_doctree_has_required_fields():
    root = HeadingNode(
        doc_id="sha256:abc", sec_id="root", level=0,
        title="Test Paper", title_bbox_norm=None, title_page=None,
        parent_sec_id=None, children_sec_ids=[], paragraph_ids=[],
        n_tokens_subtree=0,
    )
    tree = DocTree(
        doc_id="sha256:abc",
        title="Test Paper",
        authors=["Jane Doe"],
        abstract_summary=None,
        root=root,
        headings={"root": root},
        paragraphs={},
        labeled_entities={},
        parse_meta={"parser": "test"},
    )
    assert tree.doc_id == "sha256:abc"
    assert tree.title == "Test Paper"
    assert tree.headings["root"].sec_id == "root"
