"""Tests for multimodal span enrichment."""
from __future__ import annotations
from unittest.mock import patch, MagicMock
import pytest
import pymupdf

from metis.core.schema import Span
from metis.core.enrich import enrich_visual_spans, _render_bbox, ENRICHABLE_KINDS


def _make_span(kind="text", text="Normal text span content here.", **kwargs):
    defaults = dict(
        span_id="p000_L0001", doc_id="sha256:test", page=0,
        bbox_pdf=(50.0, 100.0, 300.0, 150.0),
        bbox_norm=(0.08, 0.13, 0.49, 0.19),
        text=text, reading_order=0,
        kind=kind, source="pymupdf4llm_page_boxes",
    )
    defaults.update(kwargs)
    return Span(**defaults)


@pytest.fixture()
def simple_pdf_bytes():
    """A 1-page PDF with some text."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((50, 120), "E = mc^2", fontsize=14)
    return doc.tobytes()


class TestEnrichableKinds:
    def test_formula_is_enrichable(self):
        assert "formula" in ENRICHABLE_KINDS

    def test_table_is_enrichable(self):
        assert "table" in ENRICHABLE_KINDS

    def test_text_is_not_enrichable(self):
        assert "text" not in ENRICHABLE_KINDS

    def test_picture_is_not_enrichable(self):
        assert "picture" not in ENRICHABLE_KINDS


class TestRenderBbox:
    def test_render_returns_pil_image(self, simple_pdf_bytes):
        from PIL import Image
        doc = pymupdf.open(stream=simple_pdf_bytes, filetype="pdf")
        img = _render_bbox(doc, page=0, bbox_pdf=(50, 100, 300, 150))
        doc.close()
        assert isinstance(img, Image.Image)
        assert img.width > 0
        assert img.height > 0


class TestEnrichVisualSpans:
    def test_text_spans_unchanged(self, simple_pdf_bytes):
        """Text spans pass through without modification."""
        spans = [_make_span(kind="text", text="Regular text content here.")]
        result = enrich_visual_spans(spans, simple_pdf_bytes)
        assert len(result) == 1
        assert result[0].text == "Regular text content here."
        assert result[0].content_source is None
        assert result[0].asset_path is None

    @patch("metis.core.enrich._get_p2t")
    def test_formula_span_enriched(self, mock_get_p2t, simple_pdf_bytes, tmp_path, monkeypatch):
        """Formula span gets LaTeX text from pix2text."""
        monkeypatch.setattr("metis.core.store.DATA_DIR", tmp_path)
        mock_p2t = MagicMock()
        mock_p2t.recognize_formula.return_value = "E = mc^2"
        mock_get_p2t.return_value = mock_p2t

        spans = [_make_span(kind="formula", text="garbled unicode", span_id="p000_L0005")]
        result = enrich_visual_spans(spans, simple_pdf_bytes)

        assert len(result) == 1
        assert result[0].text == "$$E = mc^2$$"
        assert result[0].content_source == "pix2text_mfr"
        assert result[0].original_text == "garbled unicode"
        assert result[0].asset_path is not None
        mock_p2t.recognize_formula.assert_called_once()

    @patch("metis.core.enrich._get_p2t")
    def test_table_span_enriched(self, mock_get_p2t, simple_pdf_bytes, tmp_path, monkeypatch):
        """Table span gets markdown text from pix2text."""
        monkeypatch.setattr("metis.core.store.DATA_DIR", tmp_path)
        mock_p2t = MagicMock()
        mock_p2t.table_ocr.recognize.return_value = {"markdown": ["| A | B |\n|---|---|\n| 1 | 2 |"]}
        mock_get_p2t.return_value = mock_p2t

        spans = [_make_span(kind="table", text="A B 1 2", span_id="p000_L0003")]
        result = enrich_visual_spans(spans, simple_pdf_bytes)

        assert len(result) == 1
        assert "| A | B |" in result[0].text
        assert result[0].content_source == "pix2text_table"
        assert result[0].original_text == "A B 1 2"

    @patch("metis.core.enrich._get_p2t")
    def test_pix2text_not_installed_returns_unchanged(self, mock_get_p2t, simple_pdf_bytes):
        """When pix2text is not installed, spans pass through unchanged."""
        mock_get_p2t.return_value = None
        spans = [_make_span(kind="formula", text="garbled")]
        result = enrich_visual_spans(spans, simple_pdf_bytes)
        assert result[0].text == "garbled"
        assert result[0].content_source is None

    @patch("metis.core.enrich._get_p2t")
    def test_pix2text_error_keeps_original_but_saves_asset(self, mock_get_p2t, simple_pdf_bytes, tmp_path, monkeypatch):
        """When pix2text raises, original text is preserved but asset is still saved."""
        monkeypatch.setattr("metis.core.store.DATA_DIR", tmp_path)
        mock_p2t = MagicMock()
        mock_p2t.recognize_formula.side_effect = RuntimeError("model failed")
        mock_get_p2t.return_value = mock_p2t

        spans = [_make_span(kind="formula", text="garbled", span_id="p000_L0005")]
        result = enrich_visual_spans(spans, simple_pdf_bytes)

        assert result[0].text == "garbled"
        assert result[0].content_source is None
        assert result[0].asset_path is not None  # image saved before extraction attempt

    @patch("metis.core.enrich._get_p2t")
    def test_mixed_spans_only_enrichable_processed(self, mock_get_p2t, simple_pdf_bytes, tmp_path, monkeypatch):
        """Only formula/table spans are processed; text and picture pass through."""
        monkeypatch.setattr("metis.core.store.DATA_DIR", tmp_path)
        mock_p2t = MagicMock()
        mock_p2t.recognize_formula.return_value = "x^2"
        mock_get_p2t.return_value = mock_p2t

        spans = [
            _make_span(kind="text", text="Regular text.", span_id="p000_L0001"),
            _make_span(kind="formula", text="garbled", span_id="p000_L0002"),
            _make_span(kind="picture", text="[[PICTURE]]", span_id="p000_L0003"),
        ]
        result = enrich_visual_spans(spans, simple_pdf_bytes)

        assert result[0].text == "Regular text."
        assert result[0].content_source is None
        assert result[1].text == "$$x^2$$"
        assert result[1].content_source == "pix2text_mfr"
        assert result[2].text == "[[PICTURE]]"
        assert result[2].content_source is None
