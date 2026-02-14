from metis.core.schema import Span
from metis.core.vectorize import _filter_embeddable

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
