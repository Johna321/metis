from metis.core.retrieve import bbox_iou, resolve_selections
from metis.core.schema import Span
from unittest.mock import patch


def test_bbox_iou_full_overlap():
    assert bbox_iou((0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0)) == 1.0


def test_bbox_iou_no_overlap():
    assert bbox_iou((0.0, 0.0, 0.5, 0.5), (0.6, 0.6, 1.0, 1.0)) == 0.0


def test_bbox_iou_partial_overlap():
    # intersection: (0.25,0.25)-(0.5,0.5) = 0.0625 area
    # union: 0.25 + 0.25 - 0.0625 = 0.4375
    iou = bbox_iou((0.0, 0.0, 0.5, 0.5), (0.25, 0.25, 0.75, 0.75))
    assert abs(iou - 0.0625 / 0.4375) < 1e-6


def test_resolve_selections_finds_overlapping_spans():
    spans = [
        Span(span_id="s1", doc_id="d", page=0,
             bbox_pdf=(0, 0, 100, 50), bbox_norm=(0.0, 0.0, 0.5, 0.5),
             text="First span", reading_order=0),
        Span(span_id="s2", doc_id="d", page=0,
             bbox_pdf=(100, 100, 200, 200), bbox_norm=(0.7, 0.7, 1.0, 1.0),
             text="Second span (no overlap)", reading_order=1),
        Span(span_id="s3", doc_id="d", page=1,
             bbox_pdf=(0, 0, 100, 50), bbox_norm=(0.0, 0.0, 0.5, 0.5),
             text="Page 1 span", reading_order=2),
    ]
    with patch("metis.core.retrieve.read_spans_jsonl", return_value=spans):
        with patch("metis.core.retrieve.paths", return_value={"spans": "fake"}):
            results = resolve_selections("d", [{"page": 0, "bbox_norm": (0.0, 0.0, 0.6, 0.6)}])
    assert len(results) == 1
    assert results[0]["span_id"] == "s1"
    assert results[0]["iou"] > 0


def test_resolve_selections_deduplicates_across_selections():
    spans = [
        Span(span_id="s1", doc_id="d", page=0,
             bbox_pdf=(0, 0, 100, 50), bbox_norm=(0.0, 0.0, 0.5, 0.5),
             text="Overlapping span", reading_order=0),
    ]
    with patch("metis.core.retrieve.read_spans_jsonl", return_value=spans):
        with patch("metis.core.retrieve.paths", return_value={"spans": "fake"}):
            results = resolve_selections("d", [
                {"page": 0, "bbox_norm": (0.0, 0.0, 0.6, 0.6)},
                {"page": 0, "bbox_norm": (0.0, 0.0, 0.4, 0.4)},
            ])
    assert len(results) == 1


def test_resolve_selections_ranked_by_iou():
    # sel = (0,0,0.5,0.5), area=0.25
    # s1 = (0.1,0.1,0.4,0.4), area=0.09, inter=(0.1,0.1)-(0.4,0.4)=0.09, union=0.25+0.09-0.09=0.25, IoU=0.36
    # s2 = (0,0,1.0,1.0), area=1.0, inter=0.25, union=0.25+1.0-0.25=1.0, IoU=0.25
    spans = [
        Span(span_id="s1", doc_id="d", page=0,
             bbox_pdf=(0, 0, 50, 50), bbox_norm=(0.1, 0.1, 0.4, 0.4),
             text="Well-fitting span", reading_order=0),
        Span(span_id="s2", doc_id="d", page=0,
             bbox_pdf=(0, 0, 200, 100), bbox_norm=(0.0, 0.0, 1.0, 1.0),
             text="Big overlap", reading_order=1),
    ]
    with patch("metis.core.retrieve.read_spans_jsonl", return_value=spans):
        with patch("metis.core.retrieve.paths", return_value={"spans": "fake"}):
            results = resolve_selections("d", [{"page": 0, "bbox_norm": (0.0, 0.0, 0.5, 0.5)}])
    assert results[0]["span_id"] == "s1"  # higher IoU (better fit)
    assert results[1]["span_id"] == "s2"
    assert results[0]["iou"] > results[1]["iou"]
