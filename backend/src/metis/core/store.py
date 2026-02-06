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
        "page_md": DATA_DIR / f"{safe}.page_md.json",
        "assets": DATA_DIR / f"{safe}_assets",
    }

def write_json(path: Path, obj) -> None:
    path.write_bytes(orjson.dumps(obj, option=orjson.OPT_INDENT_2))

def write_spans_jsonl(path: Path, spans: Iterable[Span]) -> None:
    with path.open("wb") as f:
        for s in spans:
            f.write(orjson.dumps(s.__dict__) + b"\n")

def read_spans_jsonl(path: Path) -> List[Span]:
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(Span)}
    spans: List[Span] = []
    for line in path.read_bytes().splitlines():
        d = orjson.loads(line)
        # tolerate missing optional fields and ignore unknown keys
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        # convert pos list back to tuple if present
        if "pos" in filtered and filtered["pos"] is not None:
            filtered["pos"] = tuple(filtered["pos"])
        # convert bbox tuples (orjson deserializes as lists)
        for bk in ("bbox_pdf", "bbox_norm"):
            if bk in filtered and filtered[bk] is not None:
                filtered[bk] = tuple(filtered[bk])
        spans.append(Span(**filtered))
    return spans

