from __future__ import annotations

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
