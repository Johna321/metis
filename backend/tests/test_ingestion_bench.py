from metis.benchmark.ingestion import bbox_iou, match_spans, ingestion_metrics


def test_bbox_iou_identical():
    assert bbox_iou((0, 0, 1, 1), (0, 0, 1, 1)) == 1.0


def test_bbox_iou_no_overlap():
    assert bbox_iou((0, 0, 0.5, 0.5), (0.6, 0.6, 1, 1)) == 0.0


def test_bbox_iou_partial_overlap():
    iou = bbox_iou((0, 0, 0.5, 0.5), (0.25, 0.25, 0.75, 0.75))
    # intersection: (0.25,0.25)-(0.5,0.5) = 0.25*0.25 = 0.0625
    # union: 0.25 + 0.25 - 0.0625 = 0.4375
    assert abs(iou - 0.0625 / 0.4375) < 1e-6


def test_match_spans_perfect():
    gold = [
        {"bbox_norm": (0, 0, 0.5, 0.5), "kind": "text", "reading_order": 0},
        {"bbox_norm": (0, 0.5, 0.5, 1.0), "kind": "table", "reading_order": 1},
    ]
    predicted = [
        {"bbox_norm": (0, 0, 0.5, 0.5), "kind": "text", "reading_order": 0},
        {"bbox_norm": (0, 0.5, 0.5, 1.0), "kind": "table", "reading_order": 1},
    ]
    matches = match_spans(gold, predicted, iou_threshold=0.5)
    assert len(matches) == 2


def test_ingestion_metrics_perfect():
    gold = [
        {"bbox_norm": (0, 0, 0.5, 0.5), "kind": "text", "reading_order": 0},
    ]
    predicted = [
        {"bbox_norm": (0, 0, 0.5, 0.5), "kind": "text", "reading_order": 0},
    ]
    metrics = ingestion_metrics(gold, predicted)
    assert metrics["mean_iou"] == 1.0
    assert metrics["layout_accuracy"] == 1.0
    assert metrics["coverage"] == 1.0
