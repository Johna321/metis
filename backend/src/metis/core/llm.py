from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterator, Protocol

from .schema import Message, ToolCall


@dataclass(frozen=True)
class ToolDef:
    name: str
    description: str
    parameters: dict


@dataclass(frozen=True)
class StreamEvent:
    kind: str  # "text_delta" | "tool_call_start" | "tool_call_delta" | "tool_call_done" | "message_done"
    text: str | None = None
    tool_call: ToolCall | None = None
    message: Message | None = None


class ChatModel(Protocol):
    def stream(
        self,
        messages: list[Message],
        tools: list[ToolDef],
        system: str,
    ) -> Iterator[StreamEvent]:
        ...


class AnthropicModel:
    def __init__(self, api_key: str, model: str, temperature: float = 0.0):
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._temperature = temperature

    def _to_anthropic_messages(self, messages: list[Message]) -> list[dict]:
        out = []
        for m in messages:
            if m.role == "user":
                out.append({"role": "user", "content": m.content})
            elif m.role == "assistant":
                if m.tool_calls:
                    content = []
                    if m.content:
                        content.append({"type": "text", "text": m.content})
                    for tc in m.tool_calls:
                        content.append({
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        })
                    out.append({"role": "assistant", "content": content})
                else:
                    out.append({"role": "assistant", "content": m.content})
            elif m.role == "tool":
                for tr in (m.tool_results or []):
                    out.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tr.tool_call_id,
                                "content": tr.content,
                            }
                        ],
                    })
        return out

    def _to_anthropic_tools(self, tools: list[ToolDef]) -> list[dict]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            }
            for t in tools
        ]

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolDef],
        system: str,
    ) -> Iterator[StreamEvent]:
        import anthropic as anthropic_module
        api_messages = self._to_anthropic_messages(messages)
        api_tools = self._to_anthropic_tools(tools)

        with self._client.messages.stream(
            model=self._model,
            max_tokens=4096,
            system=system,
            messages=api_messages,
            tools=api_tools if api_tools else anthropic_module.NOT_GIVEN,
            temperature=self._temperature,
        ) as stream:
            current_tool_name = None
            current_tool_id = None
            accumulated_json = ""

            for event in stream:
                if event.type == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        current_tool_name = block.name
                        current_tool_id = block.id
                        accumulated_json = ""
                        yield StreamEvent(kind="tool_call_start", text=block.name)
                elif event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        yield StreamEvent(kind="text_delta", text=delta.text)
                    elif delta.type == "input_json_delta":
                        accumulated_json += delta.partial_json
                        yield StreamEvent(kind="tool_call_delta", text=delta.partial_json)
                elif event.type == "content_block_stop":
                    if current_tool_name is not None:
                        args = json.loads(accumulated_json) if accumulated_json else {}
                        yield StreamEvent(
                            kind="tool_call_done",
                            tool_call=ToolCall(
                                id=current_tool_id,
                                name=current_tool_name,
                                arguments=args,
                            ),
                        )
                        current_tool_name = None
                        current_tool_id = None
                        accumulated_json = ""

            # Assemble final message from the completed response
            response = stream.get_final_message()
            text_parts = []
            tool_calls = []
            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_calls.append(ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input,
                    ))
            final_msg = Message(
                role="assistant",
                content="".join(text_parts) if text_parts else None,
                tool_calls=tool_calls if tool_calls else None,
            )
            yield StreamEvent(kind="message_done", message=final_msg)
