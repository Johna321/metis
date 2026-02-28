from metis.core.llm import ToolDef, StreamEvent
from metis.core.schema import ToolCall, Message
from metis.core.llm import AnthropicModel
from metis.core.llm import OpenAIModel


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


def test_openai_model_converts_messages():
    model = OpenAIModel(api_key="test-key", model="gpt-4o")
    msg = Message(role="user", content="Hello")
    converted = model._to_openai_messages([msg], system="You are helpful")
    assert converted == [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"},
    ]


def test_openai_model_converts_tool_defs():
    model = OpenAIModel(api_key="test-key", model="gpt-4o")
    td = ToolDef(
        name="web_search",
        description="Search the web",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )
    converted = model._to_openai_tools([td])
    assert converted == [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ]


def test_openai_model_converts_assistant_with_tool_calls():
    model = OpenAIModel(api_key="test-key", model="gpt-4o")
    tc = ToolCall(id="call_1", name="rag_retrieve", arguments={"query": "methods"})
    msg = Message(role="assistant", content="Let me search.", tool_calls=[tc])
    converted = model._to_openai_messages([msg], system="sys")
    # First message is system
    assert converted[0] == {"role": "system", "content": "sys"}
    # Second is the assistant message
    asst = converted[1]
    assert asst["role"] == "assistant"
    assert asst["content"] == "Let me search."
    assert len(asst["tool_calls"]) == 1
    assert asst["tool_calls"][0]["id"] == "call_1"
    assert asst["tool_calls"][0]["type"] == "function"
    assert asst["tool_calls"][0]["function"]["name"] == "rag_retrieve"


def test_openai_model_converts_tool_results():
    model = OpenAIModel(api_key="test-key", model="gpt-4o")
    from metis.core.schema import ToolResult
    tr = ToolResult(tool_call_id="call_1", content='[{"text": "found it"}]')
    msg = Message(role="tool", tool_results=[tr])
    converted = model._to_openai_messages([msg], system="sys")
    # System + tool result
    assert converted[1]["role"] == "tool"
    assert converted[1]["tool_call_id"] == "call_1"
    assert converted[1]["content"] == '[{"text": "found it"}]'
