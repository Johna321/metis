from __future__ import annotations
from typing import List
import numpy as np
import orjson
from .schema import Span, Evidence
from .store import paths, read_spans_jsonl, write_json
from ..settings import MIN_CHARS, EMBED_MODEL, TOPK_EVIDENCE

_SKIP_KINDS = {"picture", "graphic"}

def _filter_embeddable(spans: List[Span]) -> List[Span]:
    out = []
    for s in spans:
        if s.is_header or s.is_footer:
            continue
        if s.kind in _SKIP_KINDS:
            continue
        if s.text.startswith("[["):
            continue
        if len(s.text) < MIN_CHARS:
            continue
        out.append(s)
    return out


def _load_model(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def vectorize_spans(doc_id: str, model_name: str | None = None) -> dict:
    model_name = model_name or EMBED_MODEL
    p = paths(doc_id)
    spans = read_spans_jsonl(p["spans"])
    embeddable = _filter_embeddable(spans)

    if not embeddable:
        return {"doc_id": doc_id, "n_embedded": 0, "model": model_name}

    model = _load_model(model_name)
    texts = [s.text for s in embeddable]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype=np.float32)

    np.save(p["embeddings"], embeddings)
    meta = {
        "model": model_name,
        "span_ids": [s.span_id for s in embeddable],
        "dim": int(embeddings.shape[1]),
    }
    write_json(p["embeddings_meta"], meta)

    return {
        "doc_id": doc_id,
        "n_embedded": len(embeddable),
        "n_skipped": len(spans) - len(embeddable),
        "model": model_name,
        "dim": meta["dim"],
    }


def retrieve_semantic(doc_id: str, query: str, *, page: int | None = None, top_k: int = TOPK_EVIDENCE, model_name: str | None = None) -> List[Evidence]:
    model_name = model_name or EMBED_MODEL
    p = paths(doc_id)

    embeddings = np.load(p["embeddings"])
    meta = orjson.loads(p["embeddings_meta"].read_bytes())
    span_ids_embedded = meta["span_ids"]

    # Build span lookup
    all_spans = read_spans_jsonl(p["spans"])
    span_by_id = {s.span_id: s for s in all_spans}

    # Embed query
    model = _load_model(model_name)
    q_vec = model.encode([query], normalize_embeddings=True)[0].astype(np.float32)

    # Cosine similarity (embeddings already L2-normalized)
    scores = embeddings @ q_vec

    # Pair with span_ids and sort
    ranked = sorted(zip(scores, span_ids_embedded), key=lambda x: x[0], reverse=True)

    # Filter by page if requested, then take top_k
    results: List[Evidence] = []
    for score, sid in ranked:
        span = span_by_id.get(sid)
        if span is None:
            continue
        if page is not None and span.page != page:
            continue
        results.append(Evidence(
            span_id=sid,
            page=span.page,
            bbox_norm=span.bbox_norm,
            text=span.text,
            score=float(score),
        ))
        if len(results) >= top_k:
            break

    return results
