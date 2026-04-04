from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Callable

from .schema import Message, ToolResult
from .llm import ChatModel, StreamEvent
from .tools import ToolRegistry
from ..settings import CITATION_MIN_SCORE
from .store import read_messages, append_message, update_conversation


def run_agent(
    model: ChatModel,
    doc_id: str,
    user_query: str,
    tools: ToolRegistry,
    system_prompt: str,
    conv_id: str | None = None,
    max_iterations: int = 10,
    on_stream: Callable[[StreamEvent], None] | None = None,
    on_tool_result: Callable[[str, dict, str], None] | None = None,
) -> Message:
    now = datetime.now(timezone.utc).isoformat()

    # Load conversation history from disk
    history_messages: list[Message] = []
    if conv_id:
        for m in read_messages(doc_id, conv_id):
            history_messages.append(Message(role=m["role"], content=m["content"]))
        # Persist user message
        append_message(doc_id, conv_id, {"role": "user", "content": user_query, "timestamp": now})

    messages: list[Message] = history_messages + [Message(role="user", content=user_query)]
    is_first_exchange = len(history_messages) == 0
    seen_span_ids: set[str] = set()
    accumulated_evidence: list[dict] = []

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
            # Persist assistant message with evidence
            if conv_id and final_message.content:
                msg = {
                    "role": "assistant",
                    "content": final_message.content,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                if accumulated_evidence:
                    msg["evidence"] = accumulated_evidence
                append_message(doc_id, conv_id, msg)

            # Generate title for first exchange
            if conv_id and is_first_exchange and final_message.content:
                _generate_title(model, doc_id, conv_id, user_query, final_message.content, on_stream)

            if on_stream is not None:
                on_stream(StreamEvent(kind="agent_done"))
            return final_message

        # Execute tool calls and append results
        tool_results = []
        for tc in final_message.tool_calls:
            result_str = tools.call(tc.name, tc.arguments)
            if on_tool_result is not None:
                on_tool_result(tc.name, tc.arguments, result_str)
            # Emit citation_data for rag_retrieve results
            if tc.name == "rag_retrieve" and on_stream is not None:
                try:
                    items = json.loads(result_str)
                    if isinstance(items, list):
                        filtered = [
                            item for item in items
                            if item.get("score", 0.0) >= CITATION_MIN_SCORE
                            and item.get("span_id") not in seen_span_ids
                        ]
                        seen_span_ids.update(item["span_id"] for item in filtered if "span_id" in item)
                        if filtered:
                            accumulated_evidence.extend(filtered)
                            on_stream(StreamEvent(kind="citation_data", evidence=filtered, tool_call_id=tc.id, tool_name=tc.name))
                except (json.JSONDecodeError, TypeError):
                    pass
            tool_results.append(ToolResult(tool_call_id=tc.id, content=result_str))

        messages.append(Message(role="tool", tool_results=tool_results))

    # Max iterations reached
    if conv_id and final_message and final_message.content:
        msg = {
            "role": "assistant",
            "content": final_message.content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if accumulated_evidence:
            msg["evidence"] = accumulated_evidence
        append_message(doc_id, conv_id, msg)
        if is_first_exchange:
            _generate_title(model, doc_id, conv_id, user_query, final_message.content, on_stream)

    if on_stream is not None:
        on_stream(StreamEvent(kind="agent_done"))
    return final_message or Message(role="assistant", content="I was unable to complete the request within the iteration limit.")


def _generate_title(
    model: ChatModel,
    doc_id: str,
    conv_id: str,
    user_query: str,
    assistant_text: str,
    on_stream: Callable[[StreamEvent], None] | None = None,
) -> None:
    """Generate a conversation title from the first exchange. Non-blocking — failure is silently ignored."""
    try:
        title_system = "Generate a short title (3-8 words) for this conversation about a research paper. Return only the title, no quotes or punctuation."
        title_messages = [
            Message(role="user", content=user_query),
            Message(role="assistant", content=assistant_text[:500]),
            Message(role="user", content="Generate a short title for the conversation above."),
        ]
        title_text = ""
        for event in model.stream(title_messages, [], title_system):
            if event.kind == "text_delta" and event.text:
                title_text += event.text

        title_text = title_text.strip().strip('"').strip("'")
        if title_text:
            update_conversation(doc_id, conv_id, title=title_text)
            if on_stream is not None:
                on_stream(StreamEvent(kind="title_update", text=title_text, tool_call_id=conv_id))
    except Exception:
        pass  # Title generation failure is not critical
