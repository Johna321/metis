"""Agent tool factories for the hierarchical retrieval pipeline.

Exposes two new tools to the agent:
    - locate(query, top_k, sec_id)        → paragraph coordinates with previews
    - read_section(sec_id, para_start, para_end, include_subsections)
                                          → order-preserving paragraph read
"""
from __future__ import annotations
import json
from typing import Callable, List, Optional, Dict, Tuple

from .llm import ToolDef
from .retrieve_v2 import retrieve_paragraphs
from .schema_tree import DocTree, ParagraphNode
from .store import paths
from .tree_store import read_tree
from ..settings import READ_SECTION_MAX_TOKENS


def make_locate_tool(doc_id: str) -> Tuple[ToolDef, Callable[..., str]]:
    def locate(query: str, top_k: int = 5, sec_id: Optional[str] = None) -> str:
        hits = retrieve_paragraphs(doc_id, query, top_k=top_k, sec_id=sec_id)
        return json.dumps(hits)

    tool_def = ToolDef(
        name="locate",
        description=(
            "Search the paper's paragraphs by meaning and keywords. "
            "Returns coordinates (para_id, sec_id, page) with short previews and labels. "
            "Use this FIRST to find relevant content. Pass sec_id to scope the search "
            "to a particular section or subtree (prefix match — sec_id='3' matches '3', '3.1', '3.2.1')."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language search query"},
                "top_k": {"type": "integer", "description": "Number of results (default 5)", "default": 5},
                "sec_id": {"type": ["string", "null"], "description": "Optional section scope (e.g., '3.2')"},
            },
            "required": ["query"],
        },
    )
    return tool_def, locate


def _walk_section(tree: DocTree, sec_id: str, include_subsections: bool) -> List[ParagraphNode]:
    """Collect paragraphs for a section (optionally recursing into children), in reading order."""
    head = tree.headings.get(sec_id)
    if head is None:
        return []
    paras: List[ParagraphNode] = [tree.paragraphs[pid] for pid in head.paragraph_ids if pid in tree.paragraphs]
    if include_subsections:
        for child_id in head.children_sec_ids:
            paras.extend(_walk_section(tree, child_id, True))
    paras.sort(key=lambda p: p.reading_order)
    return paras


def make_read_section_tool(doc_id: str) -> Tuple[ToolDef, Callable[..., str]]:
    _tree_cache: Dict[str, DocTree] = {}

    def _get_tree() -> DocTree:
        if doc_id not in _tree_cache:
            _tree_cache[doc_id] = read_tree(paths(doc_id)["tree"])
        return _tree_cache[doc_id]

    def read_section(
        sec_id: str,
        para_start: Optional[int] = None,
        para_end: Optional[int] = None,
        include_subsections: bool = False,
    ) -> str:
        tree = _get_tree()
        head = tree.headings.get(sec_id)
        if head is None:
            return json.dumps({"error": f"Unknown sec_id: {sec_id}"})

        all_paras = _walk_section(tree, sec_id, include_subsections)

        if not include_subsections and (para_start is not None or para_end is not None):
            start = para_start or 0
            end = para_end if para_end is not None else len(all_paras)
            paras = all_paras[start:end]
        else:
            paras = all_paras

        out_paragraphs = []
        total_tokens = 0
        truncated = False
        for p in paras:
            if total_tokens + p.n_tokens > READ_SECTION_MAX_TOKENS:
                truncated = True
                break
            out_paragraphs.append({
                "para_id": p.para_id,
                "para_idx": p.para_idx,
                "page": p.page,
                "label": p.label,
                "text": p.text,
            })
            total_tokens += p.n_tokens

        if truncated and out_paragraphs:
            last = out_paragraphs[-1]
            last["text"] = last["text"] + " ... [truncated, request a narrower range]"

        return json.dumps({
            "sec_id": sec_id,
            "sec_title": head.title,
            "paragraphs": out_paragraphs,
        })

    tool_def = ToolDef(
        name="read_section",
        description=(
            "Read a section (or a range of paragraphs within it) in reading order. "
            "Use AFTER locate when you need full context around a hit, or directly "
            "when you know which section to read from the TOC. Prefer narrow ranges; "
            "only use include_subsections=True for broad questions."
        ),
        parameters={
            "type": "object",
            "properties": {
                "sec_id": {"type": "string", "description": "Section id from the TOC (e.g., '3.2')"},
                "para_start": {"type": ["integer", "null"], "description": "Start paragraph index (default 0)"},
                "para_end": {"type": ["integer", "null"], "description": "End paragraph index exclusive (default: end of section)"},
                "include_subsections": {"type": "boolean", "description": "Recurse into child sections", "default": False},
            },
            "required": ["sec_id"],
        },
    )
    return tool_def, read_section
