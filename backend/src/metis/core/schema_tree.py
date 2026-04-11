"""Hierarchical document schema: DocTree with HeadingNode and ParagraphNode.

Replaces the flat `Span` list from `schema.py`. Coordinate addressing via
(doc_id, sec_id, para_idx) with canonical `para_id` keys.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict

from .schema import BBox  # reuse the existing tuple alias


@dataclass(frozen=True)
class HeadingNode:
    """A section or sub-section heading in the document tree."""
    doc_id: str
    sec_id: str                          # dotted path ("3.2") or fixed id ("root", "abstract", "front", "refs")
    level: int                           # 0 = root, 1 = top-level section, 2 = subsection, ...
    title: str
    title_bbox_norm: Optional[BBox]      # None for synthetic root
    title_page: Optional[int]
    parent_sec_id: Optional[str]         # None only for root
    children_sec_ids: List[str]          # direct sub-headings (not paragraphs)
    paragraph_ids: List[str]             # direct paragraphs under this heading, in order
    n_tokens_subtree: int                # rolled-up token count of all descendants


@dataclass(frozen=True)
class ParagraphNode:
    """A paragraph (or other leaf content: formula, table, figure, caption, list-item)."""
    doc_id: str
    sec_id: str                          # parent section's sec_id
    para_idx: int                        # 0-indexed position within parent section
    para_id: str                         # canonical id: f"{doc_id}::{sec_id}::p{para_idx}"
    kind: str                            # "text" | "formula" | "table" | "figure" | "caption" | "list-item"
    text: str                            # cleaned, possibly enriched content
    label: Optional[str]                 # "Equation 5", "Figure 2", "Table 1" — parsed label
    bbox_norm: BBox
    page: int
    reading_order: int                   # global reading order across the document
    n_tokens: int
    # --- multimodal enrichment fields (ported from Span) ---
    asset_path: Optional[str] = None     # relative path to rendered bbox image
    content_source: Optional[str] = None # "pix2text_mfr", "pix2text_table", etc.
    original_text: Optional[str] = None  # pre-enrichment text (preserved for label matching)


@dataclass(frozen=True)
class DocTree:
    """The full hierarchical structure of an ingested document."""
    doc_id: str
    title: str
    authors: List[str]
    abstract_summary: Optional[str]      # first ~1 sentence of abstract, or None
    root: HeadingNode                    # implicit root (level=0, sec_id="root")
    headings: Dict[str, HeadingNode]     # sec_id -> HeadingNode
    paragraphs: Dict[str, ParagraphNode] # para_id -> ParagraphNode
    labeled_entities: Dict[str, str]     # normalized label -> para_id
    parse_meta: Dict                     # {"parser": ..., "level_repairs": N, ...}


def make_para_id(doc_id: str, sec_id: str, para_idx: int) -> str:
    """Build the canonical para_id for a ParagraphNode."""
    return f"{doc_id}::{sec_id}::p{para_idx}"
