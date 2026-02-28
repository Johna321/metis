from __future__ import annotations

from typing import Callable

from .schema import Message, ToolResult
from .llm import ChatModel, StreamEvent
from .tools import ToolRegistry

def run_agent(
    model: ChatModel,
    user_query: str,
    tools: ToolRegistry,
    system_prompt: str,
    max_iterations: int = 10,
    on_stream: Callable[[StreamEvent], None] | None = None,
    on_tool_result: Callable[[str, dict, str], None] | None = None,
) -> Message:
    messages: list[Message] = [Message(role="user", content=user_query)]

    for _ in range(max_iterations):
        final_message: Message | None = None

        for event in model.stream(messages, tools.tool_defs(), system_prompt):
            if on_stream is not None:
                on_stream(event)
            if event.kind == "message_done":
                final_message = event.message

        if final_message is None:
            break

        messages.append(final_message)

        # If no tool calls, we have our final answer
        if not final_message.tool_calls:
            return final_message

        # Execute tool calls and append results
        tool_results = []
        for tc in final_message.tool_calls:
            result_str = tools.call(tc.name, tc.arguments)
            if on_tool_result is not None:
                on_tool_result(tc.name, tc.arguments, result_str)
            tool_results.append(ToolResult(tool_call_id=tc.id, content=result_str))

        messages.append(Message(role="tool", tool_results=tool_results))

    # Max iterations reached, return whatever we have
    return final_message or Message(role="assistant", content="I was unable to complete the request within the iteration limit.")
