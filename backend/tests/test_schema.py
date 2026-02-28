from metis.core.schema import ToolCall, ToolResult, Message


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
