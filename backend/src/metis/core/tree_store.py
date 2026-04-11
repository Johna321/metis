"""I/O for DocTree and ParagraphNode sidecar files.

.tree.json       — full DocTree structure (nested, indexed by sec_id and para_id)
.paragraphs.jsonl — flat list of ParagraphNode in reading order, for fast iteration
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List

import orjson

from .schema_tree import DocTree, HeadingNode, ParagraphNode


def _paragraph_to_dict(p: ParagraphNode) -> dict:
    return asdict(p)


def _paragraph_from_dict(d: dict) -> ParagraphNode:
    # orjson deserializes tuples as lists — convert bbox back
    if "bbox_norm" in d and d["bbox_norm"] is not None:
        d["bbox_norm"] = tuple(d["bbox_norm"])
    return ParagraphNode(**d)


def _heading_to_dict(h: HeadingNode) -> dict:
    return asdict(h)


def _heading_from_dict(d: dict) -> HeadingNode:
    if "title_bbox_norm" in d and d["title_bbox_norm"] is not None:
        d["title_bbox_norm"] = tuple(d["title_bbox_norm"])
    return HeadingNode(**d)


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
