"""Parser protocol: PDF bytes → DocTree.

Implementations must produce a valid DocTree regardless of input quality.
The schema is the contract; parsers are interchangeable.
"""
from __future__ import annotations
from typing import Protocol

from ..schema_tree import DocTree


class Parser(Protocol):
    """A parser turns PDF bytes into a DocTree."""

    name: str

    def parse(self, pdf_bytes: bytes, doc_id: str) -> DocTree: ...
