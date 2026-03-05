from __future__ import annotations

import json
from typing import Any, Callable

import orjson
from .llm import ToolDef
from .vectorize import retrieve_hybrid
from .store import paths
from tavily import TavilyClient


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
        try:
            return fn(**arguments)
        except Exception as exc:
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


def make_rag_retrieve_tool(doc_id: str) -> tuple[ToolDef, Callable[..., str]]:
    def rag_retrieve(query: str, top_k: int = 5) -> str:
        evidence = retrieve_hybrid(doc_id=doc_id, query=query, top_k=top_k)
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

def make_web_search_tool(api_key: str) -> tuple[ToolDef, Callable[..., str]]:
    client = TavilyClient(api_key=api_key)

    def web_search(query: str, max_results: int = 5) -> str:
        response = client.search(query=query, max_results=max_results)
        return json.dumps(
            [
                {
                    "title": r['title'],
                    "url": r['url'],
                    "snippet": r['content'],
                }
                for r in response['results']
            ]
        )

    tool_def = ToolDef(
        name="web_search",
        description=(
            "Search the internet for information related to the research paper. "
            "Use for background context, related work, or current state of the art."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    )
    return tool_def, web_search

def make_read_page_tool(doc_id: str) -> tuple[ToolDef, Callable[..., str]]:
    p = paths(doc_id)
    page_md = orjson.loads(p["page_md"].read_bytes())

    def read_page(page: int) -> str:
        key = str(page)
        if key not in page_md:
            max_page = max(int(k) for k in page_md) if page_md else 0
            return json.dumps({"error": f"Page {page} not found. Valid range: 0-{max_page}"})
        return page_md[key]

    tool_def = ToolDef(
        name="read_page",
        description=(
            "Read the full text of a specific page from the paper. Use for titles, "
            "authors, abstracts, tables, figures, or any content on a known page."
        ),
        parameters={
            "type": "object",
            "properties": {
                "page": {
                    "type": "integer",
                    "description": "Page number (0-indexed)",
                },
            },
            "required": ["page"],
        },
    )
    return tool_def, read_page
