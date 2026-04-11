"""I/O for DocTree and ParagraphNode sidecar files.

.tree.json       — full DocTree structure (nested, indexed by sec_id and para_id)
.paragraphs.jsonl — flat list of ParagraphNode in reading order, for fast iteration
"""
from __future__ import annotations
import dataclasses as _dc
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List

import orjson

from .schema_tree import DocTree, HeadingNode, ParagraphNode


# Cached at module load so we don't pay reflection cost on every call.
# Filtering unknown keys gives us forward compatibility: old readers can
# tolerate new files that added fields they don't know about.
_PARA_FIELDS = {f.name for f in _dc.fields(ParagraphNode)}
_HEAD_FIELDS = {f.name for f in _dc.fields(HeadingNode)}


def _paragraph_to_dict(p: ParagraphNode) -> dict:
    return asdict(p)


def _paragraph_from_dict(d: dict) -> ParagraphNode:
    # Drop unknown keys for forward compatibility, then restore bbox tuple
    # (orjson deserializes tuples as lists).
    filtered = {k: v for k, v in d.items() if k in _PARA_FIELDS}
    if filtered.get("bbox_norm") is not None:
        filtered["bbox_norm"] = tuple(filtered["bbox_norm"])
    return ParagraphNode(**filtered)


def _heading_to_dict(h: HeadingNode) -> dict:
    return asdict(h)


def _heading_from_dict(d: dict) -> HeadingNode:
    filtered = {k: v for k, v in d.items() if k in _HEAD_FIELDS}
    if filtered.get("title_bbox_norm") is not None:
        filtered["title_bbox_norm"] = tuple(filtered["title_bbox_norm"])
    return HeadingNode(**filtered)


def write_tree(path: Path, tree: DocTree) -> None:
    """Serialize a DocTree to a .tree.json file."""
    obj = {
        "doc_id": tree.doc_id,
        "title": tree.title,
        "authors": list(tree.authors),
        "abstract_summary": tree.abstract_summary,
        "root_sec_id": tree.root.sec_id,
        "headings": {sec_id: _heading_to_dict(h) for sec_id, h in tree.headings.items()},
        "paragraphs": {pid: _paragraph_to_dict(p) for pid, p in tree.paragraphs.items()},
        "labeled_entities": dict(tree.labeled_entities),
        "parse_meta": tree.parse_meta,
    }
    path.write_bytes(orjson.dumps(obj, option=orjson.OPT_INDENT_2))


def read_tree(path: Path) -> DocTree:
    """Deserialize a DocTree from a .tree.json file."""
    obj = orjson.loads(path.read_bytes())
    headings = {sec_id: _heading_from_dict(d) for sec_id, d in obj["headings"].items()}
    paragraphs = {pid: _paragraph_from_dict(d) for pid, d in obj["paragraphs"].items()}
    return DocTree(
        doc_id=obj["doc_id"],
        title=obj["title"],
        authors=list(obj.get("authors", [])),
        abstract_summary=obj.get("abstract_summary"),
        root=headings[obj["root_sec_id"]],
        headings=headings,
        paragraphs=paragraphs,
        labeled_entities=dict(obj.get("labeled_entities", {})),
        parse_meta=dict(obj.get("parse_meta", {})),
    )


def write_paragraphs(path: Path, paragraphs: Iterable[ParagraphNode]) -> None:
    """Serialize paragraphs to a .paragraphs.jsonl file (one per line, in given order)."""
    with path.open("wb") as f:
        for p in paragraphs:
            f.write(orjson.dumps(_paragraph_to_dict(p)) + b"\n")


def read_paragraphs(path: Path) -> List[ParagraphNode]:
    """Deserialize paragraphs from a .paragraphs.jsonl file."""
    out: List[ParagraphNode] = []
    for line in path.read_bytes().splitlines():
        if not line.strip():
            continue
        out.append(_paragraph_from_dict(orjson.loads(line)))
    return out
