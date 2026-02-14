from __future__ import annotations
from typing import List
import numpy as np
from .schema import Span
from .store import paths, read_spans_jsonl, write_json
from ..settings import MIN_CHARS, EMBED_MODEL

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
