from __future__ import annotations

import json
from typing import Any, Callable

from .llm import ToolDef
from .vectorize import retrieve_semantic


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, tuple[ToolDef, Callable[..., str]]] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict,
        fn: Callable[..., str],
    ) -> None:
        self._tools[name] = (
            ToolDef(name=name, description=description, parameters=parameters),
            fn,
        )

    def tool_defs(self) -> list[ToolDef]:
        return [td for td, _ in self._tools.values()]

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        if name not in self._tools:
            return json.dumps({"error": f"Unknown tool: {name}"})
        _, fn = self._tools[name]
        return fn(**arguments)


def make_rag_retrieve_tool(doc_id: str) -> tuple[ToolDef, Callable[..., str]]:
    def rag_retrieve(query: str, top_k: int = 5) -> str:
        evidence = retrieve_semantic(doc_id=doc_id, query=query, top_k=top_k)
        return json.dumps(
            [
                {
                    "text": e.text,
                    "page": e.page,
                    "score": e.score,
                    "bbox_norm": e.bbox_norm,
                }
                for e in evidence
            ]
        )

    tool_def = ToolDef(
        name="rag_retrieve",
        description=(
            "Search the current research paper for relevant passages. "
            "Returns text excerpts with page numbers, relevance scores, and bounding boxes."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query about the paper content",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    )
    return tool_def, rag_retrieve
