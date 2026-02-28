from metis.core.llm import ToolDef, StreamEvent
from metis.core.schema import ToolCall, Message
from metis.core.llm import AnthropicModel


def test_tool_def_creation():
    td = ToolDef(
        name="rag_retrieve",
        description="Search the paper",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
    )
    assert td.name == "rag_retrieve"


def test_stream_event_text_delta():
    ev = StreamEvent(kind="text_delta", text="Hello")
    assert ev.kind == "text_delta"
    assert ev.text == "Hello"
    assert ev.tool_call is None


def test_stream_event_message_done():
    msg = Message(role="assistant", content="Final answer")
    ev = StreamEvent(kind="message_done", message=msg)
    assert ev.message.content == "Final answer"


def test_anthropic_model_converts_messages():
    """Test that AnthropicModel translates internal Messages to Anthropic format."""
    model = AnthropicModel(api_key="test-key", model="claude-sonnet-4-20250514")
    msg = Message(role="user", content="Hello")
    converted = model._to_anthropic_messages([msg])
    assert converted == [{"role": "user", "content": "Hello"}]


def test_anthropic_model_converts_tool_defs():
    """Test that AnthropicModel translates ToolDefs to Anthropic tool schema."""
    model = AnthropicModel(api_key="test-key", model="claude-sonnet-4-20250514")
    td = ToolDef(
        name="rag_retrieve",
        description="Search the paper",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )
    converted = model._to_anthropic_tools([td])
    assert converted == [
        {
            "name": "rag_retrieve",
            "description": "Search the paper",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }
    ]


def test_anthropic_model_converts_assistant_with_tool_calls():
    """Test assistant message with tool calls converts to content blocks."""
    model = AnthropicModel(api_key="test-key", model="claude-sonnet-4-20250514")
    tc = ToolCall(id="tc_1", name="rag_retrieve", arguments={"query": "methods"})
    msg = Message(role="assistant", content="Let me search.", tool_calls=[tc])
    converted = model._to_anthropic_messages([msg])
    assert len(converted) == 1
    assert converted[0]["role"] == "assistant"
    content = converted[0]["content"]
    assert len(content) == 2
    assert content[0] == {"type": "text", "text": "Let me search."}
    assert content[1] == {
        "type": "tool_use",
        "id": "tc_1",
        "name": "rag_retrieve",
        "input": {"query": "methods"},
    }


def test_anthropic_model_converts_tool_results():
    """Test tool results convert to user message with tool_result blocks."""
    model = AnthropicModel(api_key="test-key", model="claude-sonnet-4-20250514")
    from metis.core.schema import ToolResult
    tr = ToolResult(tool_call_id="tc_1", content='[{"text": "found it"}]')
    msg = Message(role="tool", tool_results=[tr])
    converted = model._to_anthropic_messages([msg])
    assert len(converted) == 1
    assert converted[0]["role"] == "user"
    assert converted[0]["content"][0]["type"] == "tool_result"
    assert converted[0]["content"][0]["tool_use_id"] == "tc_1"
