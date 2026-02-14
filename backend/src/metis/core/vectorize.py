from __future__ import annotations
from typing import List
from .schema import Span
from ..settings import MIN_CHARS

_SKIP_KINDS = {"picture", "graphic"}

def _filter_embeddable(spans: List[Span]) -> List[Span]:
    out = []
    for s in spans:
        if s.is_header or s.is_footer:
            continue
        if s.kind in _SKIP_KINDS:
            continue
        if s.text.startswith("[["):
            continue
        if len(s.text) < MIN_CHARS:
            continue
        out.append(s)
    return out
