import numpy as np
import orjson
from metis.core.schema import Span
from metis.core.vectorize import _filter_embeddable, vectorize_spans, retrieve_semantic, _get_bm25_index, _bm25_retrieve, _rrf_fuse, _mmr_rerank, retrieve_hybrid
from metis.core.store import paths, write_spans_jsonl, write_json

def _make_span(text="Hello world, this is a test span.", **kwargs):
    defaults = dict(
        span_id="p000_b000", doc_id="sha256:test", page=0,
        bbox_pdf=(0,0,1,1), bbox_norm=(0,0,1,1),
        text=text, reading_order=0,
    )
    defaults.update(kwargs)
    return Span(**defaults)

def test_filter_keeps_normal_text():
    spans = [_make_span()]
    assert len(_filter_embeddable(spans)) == 1

def test_filter_removes_pictures():
    spans = [_make_span(kind="picture", text="[[PICTURE]]")]
    assert len(_filter_embeddable(spans)) == 0

def test_filter_removes_graphics():
    spans = [_make_span(kind="graphic", text="[[GRAPHIC]]")]
    assert len(_filter_embeddable(spans)) == 0

def test_filter_removes_headers():
    spans = [_make_span(is_header=True)]
    assert len(_filter_embeddable(spans)) == 0

def test_filter_removes_footers():
    spans = [_make_span(is_footer=True)]
    assert len(_filter_embeddable(spans)) == 0

def test_filter_removes_short_text():
    spans = [_make_span(text="short")]
    assert len(_filter_embeddable(spans)) == 0

def test_filter_removes_placeholder_text():
    spans = [_make_span(text="[[TABLE 3x4]]")]
    assert len(_filter_embeddable(spans)) == 0


def _setup_doc(tmp_path, monkeypatch, spans):
    """Write spans to a temp data dir and patch DATA_DIR."""
    monkeypatch.setattr("metis.core.store.DATA_DIR", tmp_path)
    doc_id = "sha256:testdoc"
    p = paths(doc_id)
    write_spans_jsonl(p["spans"], spans)
    write_json(p["doc"], {"doc_id": doc_id, "n_pages": 1, "n_spans": len(spans)})
    return doc_id, p


def test_vectorize_creates_files(tmp_path, monkeypatch):
    spans = [_make_span(span_id=f"s{i}", text=f"This is test span number {i} with enough text.") for i in range(3)]
    doc_id, p = _setup_doc(tmp_path, monkeypatch, spans)
    meta = vectorize_spans(doc_id)
    assert p["embeddings"].exists()
    assert p["embeddings_meta"].exists()
    emb = np.load(p["embeddings"])
    assert emb.shape[0] == 3
    assert emb.dtype == np.float32
    meta_data = orjson.loads(p["embeddings_meta"].read_bytes())
    assert meta_data["span_ids"] == ["s0", "s1", "s2"]
    assert meta_data["dim"] == emb.shape[1]


def test_vectorize_filters_non_embeddable(tmp_path, monkeypatch):
    spans = [
        _make_span(span_id="good", text="This is a perfectly good text span to embed."),
        _make_span(span_id="pic", kind="picture", text="[[PICTURE]]"),
    ]
    doc_id, p = _setup_doc(tmp_path, monkeypatch, spans)
    meta = vectorize_spans(doc_id)
    emb = np.load(p["embeddings"])
    assert emb.shape[0] == 1
    meta_data = orjson.loads(p["embeddings_meta"].read_bytes())
    assert meta_data["span_ids"] == ["good"]


def test_retrieve_semantic_returns_evidence(tmp_path, monkeypatch):
    spans = [
        _make_span(span_id="s0", text="The transformer architecture uses self-attention mechanisms."),
        _make_span(span_id="s1", text="Stochastic gradient descent optimizes the loss function."),
        _make_span(span_id="s2", text="Attention allows the model to focus on relevant tokens."),
    ]
    doc_id, p = _setup_doc(tmp_path, monkeypatch, spans)
    vectorize_spans(doc_id)
    results = retrieve_semantic(doc_id, "How does attention work?")
    assert len(results) > 0
    assert all(hasattr(r, "score") for r in results)
    # attention-related spans should rank higher
    span_ids = [r.span_id for r in results]
    assert span_ids[0] in ("s0", "s2")

def test_retrieve_semantic_page_filter(tmp_path, monkeypatch):
    spans = [
        _make_span(span_id="p0", page=0, text="The transformer architecture uses self-attention mechanisms."),
        _make_span(span_id="p1", page=1, text="Attention allows the model to focus on relevant tokens."),
    ]
    doc_id, p = _setup_doc(tmp_path, monkeypatch, spans)
    vectorize_spans(doc_id)
    results = retrieve_semantic(doc_id, "attention", page=1)
    assert all(r.page == 1 for r in results)

def test_retrieve_semantic_respects_top_k(tmp_path, monkeypatch):
    spans = [_make_span(span_id=f"s{i}", text=f"This is span number {i} about neural networks and deep learning.") for i in range(10)]
    doc_id, p = _setup_doc(tmp_path, monkeypatch, spans)
    vectorize_spans(doc_id)
    results = retrieve_semantic(doc_id, "neural networks", top_k=3)
    assert len(results) <= 3

def test_bm25_index_caches(tmp_path, monkeypatch):
    spans = [
        _make_span(span_id="s0", text="The transformer architecture uses self-attention mechanisms."),
        _make_span(span_id="s1", text="Stochastic gradient descent optimizes the loss function."),
    ]
    doc_id, p = _setup_doc(tmp_path, monkeypatch, spans)
    embeddable = [s for s in spans]  # all are embeddable

    bm25_1, ids_1 = _get_bm25_index(doc_id, embeddable)
    bm25_2, ids_2 = _get_bm25_index(doc_id, embeddable)
    assert bm25_1 is bm25_2  # same object, cached


def test_bm25_retrieve_ranks_by_keyword(tmp_path, monkeypatch):
    spans = [
        _make_span(span_id="s0", text="The transformer architecture uses self-attention mechanisms."),
        _make_span(span_id="s1", text="Stochastic gradient descent optimizes the loss function."),
        _make_span(span_id="s2", text="Attention allows the model to focus on relevant tokens."),
    ]
    doc_id, p = _setup_doc(tmp_path, monkeypatch, spans)

    ranked = _bm25_retrieve(doc_id, "transformer attention", spans)
    # transformer/attention spans should rank above SGD span
    top_ids = [sid for sid, _ in ranked[:2]]
    assert "s0" in top_ids or "s2" in top_ids

def test_rrf_fuse_combines_rankings():
    # Dense ranking: A > B > C
    dense_ranked = [("A", 0.9), ("B", 0.7), ("C", 0.5)]
    # BM25 ranking: C > A > B
    bm25_ranked = [("C", 5.0), ("A", 3.0), ("B", 1.0)]

    fused = _rrf_fuse(dense_ranked, bm25_ranked, rrf_k=60)
    fused_ids = [sid for sid, _ in fused]

    # A appears high in both -> should be #1
    assert fused_ids[0] == "A"
    # All three should be present
    assert set(fused_ids) == {"A", "B", "C"}

def test_rrf_fuse_handles_disjoint():
    dense_ranked = [("A", 0.9), ("B", 0.7)]
    bm25_ranked = [("C", 5.0), ("D", 3.0)]

    fused = _rrf_fuse(dense_ranked, bm25_ranked, rrf_k=60)
    fused_ids = [sid for sid, _ in fused]

    assert set(fused_ids) == {"A", "B", "C", "D"}

def test_mmr_rerank_pure_relevance():
    """With lambda=1.0, MMR should return in relevance order."""
    candidates = [("A", 0.9), ("B", 0.8), ("C", 0.7), ("D", 0.6)]
    # Dummy embeddings: 4 candidates, 3 dims
    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    query_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    id_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3}

    result = _mmr_rerank(candidates, embeddings, query_vec, id_to_idx, top_k=3, mmr_lambda=1.0)
    # Pure relevance: A > B > C
    assert [sid for sid, _ in result] == ["A", "B", "C"]


def test_mmr_rerank_diversity():
    """With lambda=0.5, MMR should prefer diverse results over similar ones."""
    candidates = [("A", 0.9), ("B", 0.85), ("C", 0.7)]
    # A and B are nearly identical, C is orthogonal
    embeddings = np.array([
        [1.0, 0.0, 0.0],   # A
        [0.99, 0.01, 0.0],  # B - almost same as A
        [0.0, 1.0, 0.0],    # C - different direction
    ], dtype=np.float32)
    query_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    id_to_idx = {"A": 0, "B": 1, "C": 2}

    result = _mmr_rerank(candidates, embeddings, query_vec, id_to_idx, top_k=3, mmr_lambda=0.5)
    result_ids = [sid for sid, _ in result]
    # A should be first (highest relevance), C should beat B (diversity)
    assert result_ids[0] == "A"
    assert result_ids[1] == "C"  # diverse over similar

def test_retrieve_hybrid_returns_evidence(tmp_path, monkeypatch):
    spans = [
        _make_span(span_id="s0", text="The transformer architecture uses self-attention mechanisms."),
        _make_span(span_id="s1", text="Stochastic gradient descent optimizes the loss function."),
        _make_span(span_id="s2", text="Attention allows the model to focus on relevant tokens."),
    ]
    doc_id, p = _setup_doc(tmp_path, monkeypatch, spans)
    vectorize_spans(doc_id)
    results = retrieve_hybrid(doc_id, "transformer attention")
    assert len(results) > 0
    assert all(hasattr(r, "score") for r in results)
    # attention-related spans should rank high
    top_ids = [r.span_id for r in results[:2]]
    assert "s0" in top_ids or "s2" in top_ids


def test_retrieve_hybrid_finds_proper_nouns(tmp_path, monkeypatch):
    """BM25 component should find exact keyword matches that dense misses."""
    spans = [
        _make_span(span_id="authors", text="Nandan Thakur, Nils Reimers, Andreas Rückle, Abhishek Srivastava, Iryna Gurevych"),
        _make_span(span_id="abstract", text="We propose a heterogeneous benchmark for information retrieval evaluation."),
        _make_span(span_id="methods", text="Our evaluation methodology uses multiple diverse datasets across different tasks."),
    ]
    doc_id, p = _setup_doc(tmp_path, monkeypatch, spans)
    vectorize_spans(doc_id)
    results = retrieve_hybrid(doc_id, "Nandan Thakur")
    # BM25 should surface the author span even though dense embeddings won't
    top_ids = [r.span_id for r in results]
    assert "authors" in top_ids


def test_retrieve_hybrid_respects_top_k(tmp_path, monkeypatch):
    spans = [_make_span(span_id=f"s{i}", text=f"This is span number {i} about neural networks and deep learning.") for i in range(10)]
    doc_id, p = _setup_doc(tmp_path, monkeypatch, spans)
    vectorize_spans(doc_id)
    results = retrieve_hybrid(doc_id, "neural networks", top_k=3)
    assert len(results) <= 3


def test_retrieve_hybrid_page_filter(tmp_path, monkeypatch):
    spans = [
        _make_span(span_id="p0", page=0, text="The transformer architecture uses self-attention mechanisms."),
        _make_span(span_id="p1", page=1, text="Attention allows the model to focus on relevant tokens."),
    ]
    doc_id, p = _setup_doc(tmp_path, monkeypatch, spans)
    vectorize_spans(doc_id)
    results = retrieve_hybrid(doc_id, "attention", page=1)
    assert all(r.page == 1 for r in results)
