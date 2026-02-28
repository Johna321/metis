import json
from metis.core.agent import run_agent
from metis.core.schema import Message, ToolCall, ToolResult
from metis.core.llm import StreamEvent, ToolDef
from metis.core.tools import ToolRegistry

class MockModel:
    """Responds with a tool call on first turn, then a text answer on second turn."""

    def __init__(self):
        self._call_count = 0

    def stream(self, message, tools, system):
        self._call_count += 1
        if self._call_count == 1:
            # First call: issue a tool call
            tc = ToolCall(id="tc_1", name="echo", arguments={"text": "hello"})
            msg = Message(role="assistant", content=None, tool_calls=[tc])
            yield StreamEvent(kind="tool_call_start", text="echo")
            yield StreamEvent(kind="tool_call_done", tool_call=tc)
            yield StreamEvent(kind="message_done", message=msg)
        else:
            # Second call: return text
            msg = Message(role="assistant", content="The answer is hello")
            yield StreamEvent(kind="text_delta", text="The answer is hello")
            yield StreamEvent(kind="message_done", message=msg)

def test_agent_loop_executes_tool_and_returns():
    model = MockModel()
    registry = ToolRegistry()
    registry.register(
        name="echo",
        description="Echo",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}},
        fn=lambda text="": text,
    )
    result = run_agent(
        model=model,
        user_query="Say hello",
        tools=registry,
        system_prompt="You are a test assistant",
        max_iterations=5,
    )
    assert result.content == "The answer is hello"
    assert model._call_count == 2


def test_agent_loop_respects_max_iterations():
    class AlwaysToolCallModel:
        def stream(self, messages, tools, system):
            tc = ToolCall(id="tc_1", name="echo", arguments={"text": "loop"})
            msg = Message(role="assistant", content=None, tool_calls=[tc])
            yield StreamEvent(kind="tool_call_done", tool_call=tc)
            yield StreamEvent(kind="message_done", message=msg)

    model = AlwaysToolCallModel()
    registry = ToolRegistry()
    registry.register(
        name="echo",
        description="Echo",
        parameters={"type": "object", "properties": {}},
        fn=lambda text="": text,
    )
    result = run_agent(
        model=model,
        user_query="Loop forever",
        tools=registry,
        system_prompt="test",
        max_iterations=3,
    )
    # Should stop after max_iterations even though model keeps issuing tool calls
    assert result is not None
