from metis.core.schema import Span, ToolCall, ToolResult, Message


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
