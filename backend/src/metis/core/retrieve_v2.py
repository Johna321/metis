"""Hybrid retrieval over ParagraphNode: BM25 + dense + RRF + MMR.

Mirrors the pipeline in vectorize.retrieve_hybrid but operates on the new
schema. The BM25, RRF, and MMR logic is reused verbatim from vectorize.py.
"""
from __future__ import annotations
import logging
from typing import List, Dict, Optional

import numpy as np
import orjson

from .schema_tree import ParagraphNode, DocTree
from .tree_store import read_tree, read_paragraphs
from .store import paths
from .embed_v2 import _load_embed_model
from .vectorize import _tokenize, _rrf_fuse, _mmr_rerank, _bm25_cache
from ..settings import TOPK_EVIDENCE, MMR_LAMBDA, LOCATE_SNIPPET_CHARS

log = logging.getLogger(__name__)


def _truncate_preview(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut + "..."


def _is_labeled_short(p: ParagraphNode) -> bool:
    """Labeled (equation/figure/table) paragraphs return full content regardless of snippet cap."""
    return p.label is not None and p.kind in ("formula", "figure", "table")


def retrieve_paragraphs(
    doc_id: str,
    query: str,
    *,
    top_k: int = TOPK_EVIDENCE,
    sec_id: Optional[str] = None,
    rrf_k: int = 60,
    mmr_lambda: Optional[float] = None,
    snippet_chars: Optional[int] = None,
) -> List[dict]:
    """Return top-k paragraphs as dict hits (para_id, sec_id, page, label, preview, score, bbox_norm)."""
    mmr_lambda = mmr_lambda if mmr_lambda is not None else MMR_LAMBDA
    snippet_chars = snippet_chars if snippet_chars is not None else LOCATE_SNIPPET_CHARS

    p = paths(doc_id)
    tree = read_tree(p["tree"])

    embeddings = np.load(p["embeddings_v2"])
    meta = orjson.loads(p["embeddings_v2_meta"].read_bytes())
    embedded_para_ids: List[str] = meta["para_ids"]

    para_by_id = dict(tree.paragraphs)

    # Filter by section scope (prefix match on sec_id)
    if sec_id:
        allowed = {pid for pid, q in para_by_id.items()
                   if q.sec_id == sec_id or q.sec_id.startswith(sec_id + ".")}
        embedded_para_ids_filt = [pid for pid in embedded_para_ids if pid in allowed]
    else:
        embedded_para_ids_filt = embedded_para_ids

    if not embedded_para_ids_filt:
        return []

    # Query encoding — use Jina v3 retrieval.query task if available
    model = _load_embed_model()
    try:
        q_vec = model.encode([query], task="retrieval.query", normalize_embeddings=True)[0].astype(np.float32)
    except TypeError:
        q_vec = model.encode([query], normalize_embeddings=True)[0].astype(np.float32)

    # Dense scores
    id_to_idx = {pid: i for i, pid in enumerate(embedded_para_ids)}
    mask = np.zeros(len(embedded_para_ids), dtype=bool)
    for pid in embedded_para_ids_filt:
        mask[id_to_idx[pid]] = True
    dense_scores = embeddings @ q_vec
    dense_ranked = sorted(
        ((embedded_para_ids[i], float(dense_scores[i])) for i in np.where(mask)[0]),
        key=lambda x: x[1], reverse=True,
    )

    # BM25 over the filtered paragraphs.
    # Avoid caching sec_id-scoped indexes (would grow unbounded); cache only unscoped case.
    from rank_bm25 import BM25Okapi
    candidates = [para_by_id[pid] for pid in embedded_para_ids_filt if pid in para_by_id]
    if sec_id is None:
        cache_key = f"{doc_id}::v2"
        if cache_key not in _bm25_cache:
            tokenized = [_tokenize(c.text) for c in candidates]
            _bm25_cache[cache_key] = (BM25Okapi(tokenized) if tokenized else None, [c.para_id for c in candidates])
        bm25, shim_ids = _bm25_cache[cache_key]
    else:
        tokenized = [_tokenize(c.text) for c in candidates]
        bm25 = BM25Okapi(tokenized) if tokenized else None
        shim_ids = [c.para_id for c in candidates]

    if bm25 is not None:
        scores = bm25.get_scores(_tokenize(query))
        bm25_ranked = sorted(zip(shim_ids, scores), key=lambda x: x[1], reverse=True)
    else:
        bm25_ranked = []

    fetch_k = top_k * 4
    fused = _rrf_fuse(dense_ranked[:fetch_k], bm25_ranked[:fetch_k], rrf_k=rrf_k)
    reranked = _mmr_rerank(fused, embeddings, q_vec, id_to_idx, top_k=top_k, mmr_lambda=mmr_lambda)

    hits: List[dict] = []
    for pid, score in reranked:
        q = para_by_id.get(pid)
        if q is None:
            continue
        preview = q.text if _is_labeled_short(q) else _truncate_preview(q.text, snippet_chars)
        hits.append({
            "para_id": q.para_id,
            "sec_id": q.sec_id,
            "sec_title": tree.headings[q.sec_id].title if q.sec_id in tree.headings else "",
            "page": q.page,
            "label": q.label,
            "preview": preview,
            "score": float(score),
            "bbox_norm": q.bbox_norm,
        })
    return hits
