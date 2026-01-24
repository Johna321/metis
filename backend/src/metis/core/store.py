from __future__ import annotations
from pathlib import Path
import hashlib, orjson
from typing import Iterable, List
from .schema import Span
from ..settings import DATA_DIR

def doc_id_from_bytes(b: bytes) -> str:
    return "sha256:" + hashlib.sha256(b).hexdigest()

def paths(doc_id: str) -> dict:
    safe = doc_id.replace(":", "_")
    return {
        "pdf": DATA_DIR / f"{safe}.pdf",
        "spans": DATA_DIR / f"{safe}.spans.jsonl",
        "doc": DATA_DIR / f"{safe}.doc.json",
    }

def write_json(path: Path, obj) -> None:
    path.write_bytes(orjson.dumps(obj, option=orjson.OPT_INDENT_2))

def write_spans_jsonl(path: Path, spans: Iterable[Span]) -> None:
    with path.open("wb") as f:
        for s in spans:
            f.write(orjson.dumps(s.__dict__) + b"\n")

def read_spans_jsonl(path: Path) -> List[Span]:
    spans: List[Span] = []
    for line in path.read_bytes().splitlines():
        d = orjson.loads(line)
        spans.append(Span(**d))
    return spans

