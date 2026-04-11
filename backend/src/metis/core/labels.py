"""Regex-based labeled entity extraction.

Finds references to equations, figures, and tables in paragraph text
(or neighboring captions) and populates ParagraphNode.label + a doc-level
lookup: normalized_label → para_id.
"""
from __future__ import annotations
import re
from dataclasses import replace
from typing import List, Tuple, Dict

from .schema_tree import ParagraphNode

# --- Label patterns ---

_EQUATION_RE = re.compile(
    r"\b(?:Eq(?:uation|\.)?|Formula)\s*\.?\s*\(?(\d+[a-zA-Z]?|[IVX]+)\)?",
    re.IGNORECASE,
)
_FIGURE_RE = re.compile(
    r"\b(?:Fig(?:ure|\.)?)\s*\.?\s*(\d+[a-zA-Z]?|[IVX]+)",
    re.IGNORECASE,
)
_TABLE_RE = re.compile(
    r"\b(?:Tab(?:le|\.)?|TBL\.?)\s*\.?\s*(\d+[a-zA-Z]?|[IVX]+)",
    re.IGNORECASE,
)


def normalize_label(label: str) -> str:
    """Canonical form: lowercase, 'equation N' / 'figure N' / 'table N'."""
    s = label.strip()
    m = _EQUATION_RE.search(s)
    if m:
        return f"equation {m.group(1).lower()}"
    m = _FIGURE_RE.search(s)
    if m:
        return f"figure {m.group(1).lower()}"
    m = _TABLE_RE.search(s)
    if m:
        return f"table {m.group(1).lower()}"
    return re.sub(r"\s+", " ", s.lower()).strip()


def _find_label_in_text(text: str, kind: str) -> str | None:
    """Return the formatted label (e.g., 'Equation 5') if text contains one matching kind."""
    if kind == "formula":
        m = _EQUATION_RE.search(text)
        return f"Equation {m.group(1)}" if m else None
    if kind == "figure":
        m = _FIGURE_RE.search(text)
        return f"Figure {m.group(1)}" if m else None
    if kind == "table":
        m = _TABLE_RE.search(text)
        return f"Table {m.group(1)}" if m else None
    return None


def extract_labels(
    paragraphs: List[ParagraphNode],
) -> Tuple[List[ParagraphNode], Dict[str, str]]:
    """Scan paragraphs, populate `label` on formula/figure/table nodes,
    return a doc-level `normalized_label → para_id` lookup.

    Sources consulted for each labeled-kind paragraph (in order):
        1. Its own original_text (pre-enrichment) or text
        2. The preceding paragraph's text
        3. The following paragraph's text
    Later matches overwrite earlier ones for the same normalized label
    (captions after figures are the common case).
    """
    updated = list(paragraphs)
    lookup: Dict[str, str] = {}

    for i, p in enumerate(updated):
        if p.kind not in ("formula", "table", "figure"):
            continue
        sources: List[str] = []
        # Prefer original_text if present (pre-enrichment text often mentions the label)
        if p.original_text:
            sources.append(p.original_text)
        sources.append(p.text)
        if i > 0:
            sources.append(updated[i - 1].text)
        if i + 1 < len(updated):
            sources.append(updated[i + 1].text)

        for src in sources:
            label = _find_label_in_text(src, p.kind)
            if label:
                updated[i] = replace(p, label=label)
                lookup[normalize_label(label)] = p.para_id
                break

    return updated, lookup
