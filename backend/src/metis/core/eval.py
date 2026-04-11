"""Evaluation harness: metrics, runner, and LLM judge.

Produces a comparison table between two retrieval systems (baseline vs new)
against a hand-labeled JSONL eval set.
"""
from __future__ import annotations
from typing import Tuple, Set, List, Dict, Callable, Optional


# --- Metrics ---

def recall_at_k(ranked: List[str], ground_truth: Set[str], k: int) -> float:
    """1.0 if any ground_truth id is in the top-k of ranked, else 0.0."""
    top = set(ranked[:k])
    return 1.0 if top & ground_truth else 0.0


def mrr(ranked: List[str], ground_truth: Set[str]) -> float:
    """Reciprocal rank of the first ground_truth hit; 0.0 if none present."""
    for i, pid in enumerate(ranked, start=1):
        if pid in ground_truth:
            return 1.0 / i
    return 0.0


def exact_match_at_1(ranked: List[str], ground_truth: Set[str]) -> float:
    """1.0 if ranked[0] is in ground_truth, else 0.0."""
    if not ranked:
        return 0.0
    return 1.0 if ranked[0] in ground_truth else 0.0


# --- Many-to-many bbox IoU mapper for baseline comparison ---

BBox = Tuple[float, float, float, float]


def _bbox_iou(a: BBox, b: BBox) -> float:
    ix0 = max(a[0], b[0]); iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2]); iy1 = min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def map_ground_truth_to_spans(
    new_gt: Dict[str, BBox],
    old_spans: Dict[str, BBox],
    iou_threshold: float = 0.5,
) -> Set[str]:
    """Map new-branch ground-truth paragraphs to old-branch span ids via many-to-many IoU.

    Any old span with IoU >= threshold against any ground-truth paragraph counts as a match.
    Returns the union of all matching span ids.
    """
    matched: Set[str] = set()
    for span_id, span_bbox in old_spans.items():
        for gt_bbox in new_gt.values():
            if _bbox_iou(span_bbox, gt_bbox) >= iou_threshold:
                matched.add(span_id)
                break
    return matched


# --- LLM judge ---

_JUDGE_PROMPT = """\
You are grading an AI assistant's answer against a reference answer.

Question: {query}

Reference answer: {reference}

Assistant answer: {assistant}

Score the assistant answer on a 3-point scale:
    0 = wrong / missing / hallucinated
    1 = partially correct
    2 = correct (semantically matches the reference, even if phrased differently)

Reply with exactly one digit (0, 1, or 2) followed by a one-sentence justification.
"""


def _default_judge_model():
    """Construct a cheap Claude model for judging. Pulls api_key from settings."""
    from .llm import AnthropicModel
    from ..settings import ANTHROPIC_API_KEY
    return AnthropicModel(
        api_key=ANTHROPIC_API_KEY,
        model="claude-haiku-4-5-20251001",
        temperature=0.0,
    )


def judge_answer(query: str, reference: str, assistant: str, model=None) -> Tuple[int, str]:
    """Use an LLM judge to score an assistant answer. Returns (score, justification)."""
    from .schema import Message

    m = model or _default_judge_model()
    prompt = _JUDGE_PROMPT.format(query=query, reference=reference, assistant=assistant)
    text = ""
    for event in m.stream([Message(role="user", content=prompt)], [], ""):
        if event.kind == "text_delta" and event.text:
            text += event.text
    text = text.strip()
    if not text:
        return 0, "empty judge response"
    score_char = text[0]
    try:
        score = int(score_char)
        if score not in (0, 1, 2):
            score = 0
    except ValueError:
        score = 0
    justification = text[1:].strip(" :.-")
    return score, justification


# --- Runner ---

def run_eval(
    eval_set_path,
    output_dir,
    system_name: str,
    agent_runner: Callable[[str, str], Tuple[str, List[str]]],
) -> dict:
    """Run the eval set against an agent runner callable.

    `agent_runner(doc_id_or_paper, query) -> (answer_text, ordered_para_ids)`
    returns the assistant's final answer text and the ordered list of para_ids
    the system retrieved (to compute retrieval metrics).
    """
    import orjson
    from pathlib import Path
    from datetime import datetime, timezone

    eval_set_path = Path(eval_set_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_path = output_dir / f"{timestamp}_{system_name}.jsonl"

    results: List[dict] = []
    with eval_set_path.open("rb") as f, out_path.open("wb") as out:
        for raw in f:
            if not raw.strip():
                continue
            q = orjson.loads(raw)
            gt = set(q.get("ground_truth_para_ids", []))
            doc_id = q.get("doc_id", q["paper"])
            try:
                answer, ranked = agent_runner(doc_id, q["query"])
            except Exception as exc:
                record = {**q, "error": str(exc), "system": system_name}
                out.write(orjson.dumps(record) + b"\n")
                results.append(record)
                continue
            r5 = recall_at_k(ranked, gt, k=5)
            r10 = recall_at_k(ranked, gt, k=10)
            mrr_ = mrr(ranked, gt)
            em1 = exact_match_at_1(ranked, gt)
            if q.get("ground_truth_answer"):
                try:
                    j_score, j_reason = judge_answer(q["query"], q["ground_truth_answer"], answer)
                except Exception as exc:
                    j_score, j_reason = 0, f"judge error: {exc}"
            else:
                j_score, j_reason = 0, ""
            record = {
                **q,
                "system": system_name,
                "answer": answer,
                "ranked": ranked[:20],
                "recall_at_5": r5,
                "recall_at_10": r10,
                "mrr": mrr_,
                "em_at_1": em1,
                "judge_score": j_score,
                "judge_reason": j_reason,
            }
            out.write(orjson.dumps(record) + b"\n")
            results.append(record)

    return {"n": len(results), "output": str(out_path)}
