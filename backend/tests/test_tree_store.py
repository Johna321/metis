"""Tests for DocTree serialization."""
from __future__ import annotations
import pytest

from metis.core.schema_tree import DocTree, HeadingNode, ParagraphNode, make_para_id
from metis.core.tree_store import write_tree, read_tree, write_paragraphs, read_paragraphs


def _sample_tree(doc_id: str = "sha256:test") -> DocTree:
    root = HeadingNode(
        doc_id=doc_id, sec_id="root", level=0, title="A Paper",
        title_bbox_norm=None, title_page=None,
        parent_sec_id=None, children_sec_ids=["1"],
        paragraph_ids=[], n_tokens_subtree=20,
    )
    sec1 = HeadingNode(
        doc_id=doc_id, sec_id="1", level=1, title="Introduction",
        title_bbox_norm=(0.1, 0.05, 0.5, 0.08), title_page=0,
        parent_sec_id="root", children_sec_ids=[],
        paragraph_ids=[make_para_id(doc_id, "1", 0)], n_tokens_subtree=20,
    )
    p0 = ParagraphNode(
        doc_id=doc_id, sec_id="1", para_idx=0,
        para_id=make_para_id(doc_id, "1", 0),
        kind="text", text="This is the intro.",
        label=None, bbox_norm=(0.1, 0.15, 0.9, 0.25),
        page=0, reading_order=0, n_tokens=5,
    )
    return DocTree(
        doc_id=doc_id, title="A Paper", authors=["X"],
        abstract_summary=None, root=root,
        headings={"root": root, "1": sec1},
        paragraphs={p0.para_id: p0},
        labeled_entities={},
        parse_meta={"parser": "test"},
    )


def test_write_and_read_tree_roundtrip(tmp_path):
    from metis.core.store import paths
    import metis.core.store as store_mod
    orig = store_mod.DATA_DIR
    store_mod.DATA_DIR = tmp_path
    try:
        tree = _sample_tree()
        p = paths(tree.doc_id)
        write_tree(p["tree"], tree)
        loaded = read_tree(p["tree"])
        assert loaded.doc_id == tree.doc_id
        assert loaded.title == tree.title
        assert set(loaded.headings.keys()) == {"root", "1"}
        assert loaded.headings["1"].children_sec_ids == []
        assert loaded.paragraphs[make_para_id(tree.doc_id, "1", 0)].text == "This is the intro."
    finally:
        store_mod.DATA_DIR = orig


def test_write_and_read_paragraphs_jsonl(tmp_path):
    from metis.core.store import paths
    import metis.core.store as store_mod
    orig = store_mod.DATA_DIR
    store_mod.DATA_DIR = tmp_path
    try:
        tree = _sample_tree()
        p = paths(tree.doc_id)
        para_list = list(tree.paragraphs.values())
        write_paragraphs(p["paragraphs"], para_list)
        loaded = read_paragraphs(p["paragraphs"])
        assert len(loaded) == 1
        assert loaded[0].bbox_norm == (0.1, 0.15, 0.9, 0.25)  # tuple restored
        assert loaded[0].kind == "text"
    finally:
        store_mod.DATA_DIR = orig


def test_read_paragraphs_tolerates_unknown_keys(tmp_path):
    """Old code loading new files: unknown keys must be silently dropped."""
    import orjson
    jsonl_path = tmp_path / "test.paragraphs.jsonl"
    entry = {
        "doc_id": "sha256:test",
        "sec_id": "1",
        "para_idx": 0,
        "para_id": "sha256:test::1::p0",
        "kind": "text",
        "text": "hello",
        "label": None,
        "bbox_norm": [0, 0, 1, 1],
        "page": 0,
        "reading_order": 0,
        "n_tokens": 1,
        "asset_path": None,
        "content_source": None,
        "original_text": None,
        # Unknown forward-compat field — must be dropped, not crash:
        "future_field_from_tomorrow": "ignore me",
    }
    jsonl_path.write_bytes(orjson.dumps(entry) + b"\n")

    from metis.core.tree_store import read_paragraphs
    loaded = read_paragraphs(jsonl_path)
    assert len(loaded) == 1
    assert loaded[0].text == "hello"
    assert loaded[0].bbox_norm == (0, 0, 1, 1)
