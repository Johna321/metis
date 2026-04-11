"""Tests for labeled entity extraction."""
from __future__ import annotations
from dataclasses import replace

from metis.core.labels import normalize_label, extract_labels
from metis.core.schema_tree import ParagraphNode, make_para_id


def _p(idx, text, kind="text", sec_id="1"):
    doc_id = "sha256:test"
    return ParagraphNode(
        doc_id=doc_id, sec_id=sec_id, para_idx=idx,
        para_id=make_para_id(doc_id, sec_id, idx),
        kind=kind, text=text, label=None,
        bbox_norm=(0, 0, 1, 1), page=0, reading_order=idx, n_tokens=len(text.split()),
    )


def test_normalize_label_variants():
    assert normalize_label("Eq. (5)") == "equation 5"
    assert normalize_label("Equation 5") == "equation 5"
    assert normalize_label("EQUATION   5") == "equation 5"
    assert normalize_label("Figure 2") == "figure 2"
    assert normalize_label("Fig. 2") == "figure 2"
    assert normalize_label("Table 1") == "table 1"
    assert normalize_label("TBL. I") == "table i"


def test_extract_labels_equation_from_neighbor_text():
    paras = [
        _p(0, "We define the attention score as shown in Eq. (5):"),
        _p(1, "$$\\text{softmax}(QK^T)$$", kind="formula"),
        _p(2, "where Q and K are query and key matrices."),
    ]
    updated, lookup = extract_labels(paras)
    assert lookup.get("equation 5") == paras[1].para_id
    formula_p = next(p for p in updated if p.kind == "formula")
    assert formula_p.label == "Equation 5"


def test_extract_labels_table_from_caption():
    paras = [
        _p(0, "Table 1: Results on benchmark X.", kind="caption"),
        _p(1, "| Method | Acc |\n| A | 90 |", kind="table"),
    ]
    updated, lookup = extract_labels(paras)
    assert lookup.get("table 1") == paras[1].para_id
    table_p = next(p for p in updated if p.kind == "table")
    assert table_p.label == "Table 1"


def test_extract_labels_figure_from_caption_below():
    paras = [
        _p(0, "[[PICTURE]]", kind="figure"),
        _p(1, "Figure 2: Architecture diagram.", kind="caption"),
    ]
    updated, lookup = extract_labels(paras)
    assert lookup.get("figure 2") == paras[0].para_id
    fig_p = next(p for p in updated if p.kind == "figure")
    assert fig_p.label == "Figure 2"
