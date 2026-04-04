from __future__ import annotations
from pathlib import Path
import hashlib, orjson, time, secrets
from datetime import datetime, timezone
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
        "embeddings": DATA_DIR / f"{safe}.embeddings.npy",
        "embeddings_meta": DATA_DIR / f"{safe}.embeddings_meta.json",
        "conversations": DATA_DIR / f"{safe}.conversations.json",
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

def conv_path(doc_id: str, conv_id: str) -> Path:
    """Path to a single conversation's JSONL message file."""
    safe = doc_id.replace(":", "_")
    return DATA_DIR / f"{safe}.conv_{conv_id}.jsonl"


def _write_conv_index(doc_id: str, conversations: list[dict]) -> None:
    """Write the conversation index JSON."""
    p = paths(doc_id)["conversations"]
    write_json(p, {"conversations": conversations})


def _read_conv_index(doc_id: str) -> list[dict]:
    """Read raw conversation index entries (no message_count)."""
    p = paths(doc_id)["conversations"]
    if not p.exists():
        return []
    data = orjson.loads(p.read_bytes())
    return data.get("conversations", [])


def read_conversations(doc_id: str) -> list[dict]:
    """Read conversation index with computed message_count. Sorted: pinned first, then by updated_at desc."""
    entries = _read_conv_index(doc_id)
    for c in entries:
        cp = conv_path(doc_id, c["id"])
        c["message_count"] = sum(1 for _ in cp.open()) if cp.exists() else 0
    entries.sort(key=lambda c: (c["pinned"], c["updated_at"]), reverse=True)
    return entries


def create_conversation(doc_id: str) -> dict:
    """Create a new conversation. Returns the new entry dict."""
    entries = _read_conv_index(doc_id)
    now = datetime.now(timezone.utc).isoformat()
    conv_id = f"conv_{int(time.time())}_{secrets.token_hex(2)}"
    entry = {
        "id": conv_id,
        "title": "New Conversation",
        "pinned": False,
        "created_at": now,
        "updated_at": now,
    }
    entries.append(entry)
    _write_conv_index(doc_id, entries)
    return {**entry, "message_count": 0}


def update_conversation(doc_id: str, conv_id: str, title: str | None = None, pinned: bool | None = None) -> dict:
    """Update a conversation's title and/or pinned status. Returns updated entry."""
    entries = _read_conv_index(doc_id)
    for c in entries:
        if c["id"] == conv_id:
            if title is not None:
                c["title"] = title
            if pinned is not None:
                c["pinned"] = pinned
            c["updated_at"] = datetime.now(timezone.utc).isoformat()
            _write_conv_index(doc_id, entries)
            cp = conv_path(doc_id, conv_id)
            c["message_count"] = sum(1 for _ in cp.open()) if cp.exists() else 0
            return c
    raise FileNotFoundError(f"Conversation not found: {conv_id}")


def delete_conversation(doc_id: str, conv_id: str) -> None:
    """Delete a conversation: remove from index and delete JSONL file."""
    entries = _read_conv_index(doc_id)
    entries = [c for c in entries if c["id"] != conv_id]
    _write_conv_index(doc_id, entries)
    cp = conv_path(doc_id, conv_id)
    if cp.exists():
        cp.unlink()


def append_message(doc_id: str, conv_id: str, message: dict) -> None:
    """Append a message to a conversation's JSONL file."""
    cp = conv_path(doc_id, conv_id)
    with cp.open("ab") as f:
        f.write(orjson.dumps(message) + b"\n")
    # Touch updated_at in the index
    entries = _read_conv_index(doc_id)
    for c in entries:
        if c["id"] == conv_id:
            c["updated_at"] = datetime.now(timezone.utc).isoformat()
            break
    _write_conv_index(doc_id, entries)


def read_messages(doc_id: str, conv_id: str) -> list[dict]:
    """Read all messages from a conversation's JSONL file."""
    cp = conv_path(doc_id, conv_id)
    if not cp.exists():
        return []
    messages = []
    for line in cp.read_bytes().splitlines():
        if line.strip():
            messages.append(orjson.loads(line))
    return messages

