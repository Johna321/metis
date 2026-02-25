from __future__ import annotations
from typing import Tuple


BBox = Tuple[float, float, float, float]


def bbox_iou(a: BBox, b: BBox) -> float:
    """Compute intersection-over-union of two bboxes (x0, y0, x1, y1)."""
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def match_spans(
    gold: list[dict],
    predicted: list[dict],
    iou_threshold: float = 0.5,
) -> list[tuple[dict, dict, float]]:
    """Match gold spans to predicted spans by best IoU. Greedy matching.

    Returns list of (gold_span, predicted_span, iou) tuples.
    """
    used = set()
    matches = []
    for g in gold:
        best_iou = 0.0
        best_idx = -1
        for j, p in enumerate(predicted):
            if j in used:
                continue
            iou = bbox_iou(g["bbox_norm"], p["bbox_norm"])
            if iou > best_iou:
                best_iou = iou
                best_idx = j
        if best_iou >= iou_threshold and best_idx >= 0:
            used.add(best_idx)
            matches.append((g, predicted[best_idx], best_iou))
    return matches


def ingestion_metrics(gold: list[dict], predicted: list[dict]) -> dict:
    """Compute ingestion quality metrics between gold and predicted spans."""
    matches = match_spans(gold, predicted)

    # Mean IoU of matched spans
    mean_iou = sum(iou for _, _, iou in matches) / len(matches) if matches else 0.0

    # Layout classification accuracy
    kind_correct = sum(1 for g, p, _ in matches if g.get("kind") == p.get("kind"))
    layout_accuracy = kind_correct / len(matches) if matches else 0.0

    # Coverage: fraction of gold spans that were matched
    coverage = len(matches) / len(gold) if gold else 0.0

    # Spurious: fraction of predicted spans that weren't matched
    spurious = 1 - (len(matches) / len(predicted)) if predicted else 0.0

    return {
        "mean_iou": round(mean_iou, 4),
        "layout_accuracy": round(layout_accuracy, 4),
        "coverage": round(coverage, 4),
        "spurious_rate": round(spurious, 4),
        "n_gold": len(gold),
        "n_predicted": len(predicted),
        "n_matched": len(matches),
    }
