"""Compact table-of-contents rendering from a DocTree, suitable for system prompts."""
from __future__ import annotations
from typing import Optional

from .schema_tree import DocTree, HeadingNode
from ..settings import TOC_MAX_DEPTH


def render_toc(tree: DocTree, max_depth: Optional[int] = None) -> str:
    """Render an indented TOC listing headings up to `max_depth` (1-indexed)."""
    depth_cap = max_depth if max_depth is not None else TOC_MAX_DEPTH
    lines = [f'# "{tree.title}"', ""]

    def walk(h: HeadingNode, depth: int) -> None:
        if depth > depth_cap:
            return
        if h.sec_id != "root":
            indent = "  " * (depth - 1)
            lines.append(f"{indent}{h.sec_id:<10} — {h.title}")
        for child_id in h.children_sec_ids:
            child = tree.headings.get(child_id)
            if child is None:
                continue
            walk(child, depth + 1)

    for top_id in tree.root.children_sec_ids:
        top = tree.headings.get(top_id)
        if top is not None:
            walk(top, 1)

    return "\n".join(lines)
