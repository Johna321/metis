import numpy as np
import orjson
from metis.core.schema import Span
from metis.core.vectorize import _filter_embeddable, vectorize_spans
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
