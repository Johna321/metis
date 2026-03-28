import orjson
from metis.core.schema import Span, ToolCall, ToolResult, Message
from metis.core.store import read_spans_jsonl, write_spans_jsonl


def test_span_multimodal_fields_default_none():
    """New multimodal fields default to None, preserving backward compat."""
    s = Span(
        span_id="p000_b000", doc_id="sha256:test", page=0,
        bbox_pdf=(0, 0, 1, 1), bbox_norm=(0, 0, 1, 1),
        text="Hello world", reading_order=0,
    )
    assert s.asset_path is None
    assert s.content_source is None
    assert s.original_text is None


def test_span_multimodal_fields_set():
    """New multimodal fields can be set explicitly."""
    s = Span(
        span_id="p000_L0010", doc_id="sha256:test", page=0,
        bbox_pdf=(0, 0, 1, 1), bbox_norm=(0, 0, 1, 1),
        text="$$E = mc^2$$", reading_order=0,
        kind="formula",
        asset_path="sha256_test_assets/images/p000_L0010.png",
        content_source="pix2text_mfr",
        original_text="𝐸 = 𝑚𝑐2",
    )
    assert s.asset_path == "sha256_test_assets/images/p000_L0010.png"
    assert s.content_source == "pix2text_mfr"
    assert s.original_text == "𝐸 = 𝑚𝑐2"


def test_spans_roundtrip_with_enrichment(tmp_path):
    """Enriched spans survive write -> read round-trip."""
    spans = [Span(
        span_id="p000_L0010", doc_id="sha256:test", page=0,
        bbox_pdf=(0.0, 0.0, 100.0, 50.0), bbox_norm=(0.0, 0.0, 0.5, 0.25),
        text="$$E = mc^2$$", reading_order=0,
        kind="formula", source="pymupdf4llm_page_boxes",
        asset_path="sha256_test_assets/images/p000_L0010.png",
        content_source="pix2text_mfr",
        original_text="𝐸 = 𝑚𝑐2",
    )]
    path = tmp_path / "test.spans.jsonl"
    write_spans_jsonl(path, spans)
    loaded = read_spans_jsonl(path)
    assert len(loaded) == 1
    s = loaded[0]
    assert s.text == "$$E = mc^2$$"
    assert s.asset_path == "sha256_test_assets/images/p000_L0010.png"
    assert s.content_source == "pix2text_mfr"
    assert s.original_text == "𝐸 = 𝑚𝑐2"
    assert s.bbox_pdf == (0.0, 0.0, 100.0, 50.0)  # tuple, not list


def test_old_spans_file_loads_with_new_schema(tmp_path):
    """JSONL written without new fields loads with None defaults."""
    old_span_json = orjson.dumps({
        "span_id": "p000_b000", "doc_id": "sha256:old", "page": 0,
        "bbox_pdf": [0, 0, 1, 1], "bbox_norm": [0, 0, 1, 1],
        "text": "Old span without enrichment fields", "reading_order": 0,
        "is_header": False, "is_footer": False,
        "kind": "text", "pos": None, "source": "pymupdf_blocks",
    })
    path = tmp_path / "old.spans.jsonl"
    path.write_bytes(old_span_json + b"\n")
    loaded = read_spans_jsonl(path)
    assert len(loaded) == 1
    assert loaded[0].asset_path is None
    assert loaded[0].content_source is None
    assert loaded[0].original_text is None


def test_tool_call_creation():
    tc = ToolCall(id="tc_1", name="rag_retrieve", arguments={"query": "hello"})
    assert tc.id == "tc_1"
    assert tc.name == "rag_retrieve"
    assert tc.arguments == {"query": "hello"}


def test_message_user():
    m = Message(role="user", content="What is this paper about?")
    assert m.role == "user"
    assert m.content == "What is this paper about?"
    assert m.tool_calls is None
    assert m.tool_results is None


def test_message_assistant_with_tool_calls():
    tc = ToolCall(id="tc_1", name="rag_retrieve", arguments={"query": "methods"})
    m = Message(role="assistant", content=None, tool_calls=[tc])
    assert m.tool_calls == [tc]
    assert m.content is None


def test_message_tool_result():
    tr = ToolResult(tool_call_id="tc_1", content='[{"text": "results"}]')
    m = Message(role="tool", content=None, tool_results=[tr])
    assert m.tool_results == [tr]
