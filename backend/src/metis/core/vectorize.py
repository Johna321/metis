from __future__ import annotations
import re
from typing import List
import numpy as np
import orjson
from .schema import Span, Evidence
from .store import paths, read_spans_jsonl, write_json
from ..settings import MIN_CHARS, EMBED_MODEL, TOPK_EVIDENCE, MMR_LAMBDA

_SKIP_KINDS = {"picture", "graphic", "formula", "table"}

def _filter_embeddable(spans: List[Span]) -> List[Span]:
    out = []
    for s in spans:
        if s.is_header or s.is_footer:
            continue
        if s.kind in _SKIP_KINDS and s.content_source is None:
            continue
        if s.text.startswith("[["):
            continue
        if len(s.text) < MIN_CHARS:
            continue
        out.append(s)
    return out


_model_cache: dict[str, object] = {}

def _load_model(model_name: str):
    if model_name not in _model_cache:
        import logging
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
        from sentence_transformers import SentenceTransformer
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]

import nltk
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

_STOP_WORDS = set(stopwords.words("english"))
_stemmer = PorterStemmer()

_LATEX_CMD_RE = re.compile(r"\\[a-zA-Z]+")

def _tokenize(text: str) -> list[str]:
    """Tokenize, lowercase, remove stopwords and non-alpha, stem. LaTeX commands survive."""
    latex_tokens = [t.lower() for t in _LATEX_CMD_RE.findall(text)]
    tokens = word_tokenize(text.lower())
    stemmed = [_stemmer.stem(t) for t in tokens if t.isalpha() and t not in _STOP_WORDS]
    return stemmed + latex_tokens

from rank_bm25 import BM25Okapi

_bm25_cache: dict[str, tuple[BM25Okapi, list[str]]] = {}

def _get_bm25_index(doc_id: str, spans: list[Span]) -> tuple[BM25Okapi | None, list[str]]:
    if doc_id not in _bm25_cache:
        if not spans:
            return None, []
        tokenized = [_tokenize(s.text) for s in spans]
        bm25 = BM25Okapi(tokenized)
        span_ids = [s.span_id for s in spans]
        _bm25_cache[doc_id] = (bm25, span_ids)
    return _bm25_cache[doc_id]

def _bm25_retrieve(doc_id: str, query: str, spans: list[Span]) -> list[tuple[str, float]]:
    bm25, span_ids = _get_bm25_index(doc_id, spans)
    if bm25 is None:
        return []
    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    ranked = sorted(zip(span_ids, scores), key=lambda x: x[1], reverse=True)
    return ranked

def _rrf_fuse(
    dense_ranked: list[tuple[str, float]],
    bm25_ranked: list[tuple[str, float]],
    rrf_k: int = 60,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for rank, (sid, _) in enumerate(dense_ranked):
        scores[sid] = scores.get(sid, 0.0) + 1.0 / (rrf_k + rank + 1)
    for rank, (sid, _) in enumerate(bm25_ranked):
        scores[sid] = scores.get(sid, 0.0) + 1.0 / (rrf_k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def _mmr_rerank(
    candidates: list[tuple[str, float]],
    embeddings: np.ndarray,
    query_vec: np.ndarray,
    id_to_idx: dict[str, int],
    top_k: int,
    mmr_lambda: float = 0.7,
) -> list[tuple[str, float]]:
    if not candidates:
        return []

    selected: list[tuple[str, float]] = []
    remaining = list(candidates)

    for _ in range(min(top_k, len(remaining))):
        best_score = -float("inf")
        best_idx = 0

        for i, (sid, relevance) in enumerate(remaining):
            emb_idx = id_to_idx.get(sid)
            if emb_idx is None:
                continue
            cand_vec = embeddings[emb_idx]

            # Max similarity to alreadys-selected spans
            max_sim = 0.0
            for sel_sid, _ in selected:
                sel_emb_idx = id_to_idx.get(sel_sid)
                if sel_emb_idx is not None:
                    sim = float(cand_vec @ embeddings[sel_emb_idx])
                    if sim > max_sim:
                        max_sim = sim

            mmr_score = mmr_lambda * relevance - (1 - mmr_lambda) * max_sim
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        selected.append(remaining.pop(best_idx))

    return selected

def retrieve_hybrid(
    doc_id: str,
    query: str,
    *,
    page: int | None = None,
    top_k: int = TOPK_EVIDENCE,
    rrf_k: int = 60,
    mmr_lambda: float | None = None,
    model_name: str | None = None,
) -> List[Evidence]:
    mmr_lambda = mmr_lambda if mmr_lambda is not None else MMR_LAMBDA
    model_name = model_name or EMBED_MODEL
    p = paths(doc_id)

    # Load embeddings and spans
    embeddings = np.load(p["embeddings"])
    meta = orjson.loads(p["embeddings_meta"].read_bytes())
    span_ids_embedded = meta["span_ids"]
    all_spans = read_spans_jsonl(p["spans"])
    span_by_id = {s.span_id: s for s in all_spans}

    # Build embeddable span lists (same set used for both dense and BM25)
    embeddable = [span_by_id[sid] for sid in span_ids_embedded if sid in span_by_id]

    # Filter by page if requested
    if page is not None:
        page_ids = {s.span_id for s in embeddable if s.page == page}
    else:
        page_ids = None

    model = _load_model(model_name)
    q_vec = model.encode([query], normalize_embeddings=True)[0].astype(np.float32)
    dense_scores = embeddings @ q_vec
    dense_ranked = sorted(
        zip(span_ids_embedded, dense_scores.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    if page_ids is not None:
        dense_ranked = [(sid, s) for sid, s in dense_ranked if sid in page_ids]

    # BM25 retrieval
    bm25_ranked = _bm25_retrieve(doc_id, query, embeddable)
    if page_ids is not None:
        bm25_ranked = [(sid, s) for sid, s in bm25_ranked if sid in page_ids]

    # RRF fusion
    fetch_k = top_k * 4
    fused = _rrf_fuse(dense_ranked[:fetch_k], bm25_ranked[:fetch_k], rrf_k=rrf_k)

    # MMR reranking
    id_to_idx = {sid: i for i, sid in enumerate(span_ids_embedded)}
    reranked = _mmr_rerank(fused, embeddings, q_vec, id_to_idx, top_k=top_k, mmr_lambda=mmr_lambda)

    # Build Evidence results
    results: List[Evidence] = []
    for sid, score in reranked:
        span = span_by_id.get(sid)
        if span is None:
            continue
        results.append(Evidence(
            span_id=sid,
            page=span.page,
            bbox_norm=span.bbox_norm,
            text=span.text,
            score=float(score),
        ))

    return results

def vectorize_spans(doc_id: str, model_name: str | None = None) -> dict:
    model_name = model_name or EMBED_MODEL
    p = paths(doc_id)

    if p["embeddings"].exists() and p["embeddings_meta"].exists():
        meta = orjson.loads(p["embeddings_meta"].read_bytes())
        embeddings = np.load(p["embeddings"])
        return {
            "doc_id": doc_id,
            "n_embedded": embeddings.shape[0],
            "n_skipped": None,
            "model": meta.get("model", model_name),
            "dim": meta["dim"],
            "was_cached": True,
        }

    spans = read_spans_jsonl(p["spans"])
    embeddable = _filter_embeddable(spans)

    if not embeddable:
        return {
            "doc_id": doc_id,
            "n_embedded": 0,
            "n_skipped": len(spans),
            "model": model_name,
            "dim": None,
            "was_cached": False,
        }

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
        "was_cached": False,
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
