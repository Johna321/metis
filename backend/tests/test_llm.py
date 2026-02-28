from metis.core.llm import ToolDef, StreamEvent
from metis.core.schema import ToolCall, Message


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
