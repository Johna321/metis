"""Embedding pipeline v2: Jina v3 + late chunking + contextual prefix.

This module contains:
    - build_context_prefix(): deterministic doc-metadata + section-path + label prefix
    - assemble_doc_text(): concatenate paragraphs in reading order, tracking char spans
    - late_chunked_embeddings(): forward whole document, mean-pool over paragraph spans
    - vectorize_tree(): orchestrate ingest-time embedding for a DocTree

The paragraph text exposed to the LLM is never the prefixed version — the prefix
is only used for the encoder input and is stripped via the offset tracking.
"""
from __future__ import annotations
import logging
from typing import Any, List, Tuple, Dict, Optional

import numpy as np

from .schema_tree import DocTree, ParagraphNode

log = logging.getLogger(__name__)


# -------------------------- prefix + assembly --------------------------

def build_context_prefix(p: ParagraphNode, tree: DocTree) -> str:
    """Build the deterministic context prefix for a paragraph.

    Format:
        Doc: "<title>" | <abstract_first_sentence>
        <section_path>[ > <label>]
    """
    title = tree.title.strip() or "Untitled"
    abstract_hint = (tree.abstract_summary or "").strip()
    line1 = f'Doc: "{title}"'
    if abstract_hint:
        line1 += f" | {abstract_hint}"

    # Walk from paragraph's section up to root
    path_parts: List[str] = []
    ancestors: List[str] = []
    cur_sec = tree.headings.get(p.sec_id)
    while cur_sec is not None and cur_sec.sec_id != "root":
        label = f"{cur_sec.sec_id} {cur_sec.title}".strip()
        ancestors.append(label)
        parent = cur_sec.parent_sec_id
        cur_sec = tree.headings.get(parent) if parent else None
    path_parts = list(reversed(ancestors))
    if p.label:
        path_parts.append(p.label)
    line2 = " > ".join(path_parts) if path_parts else ""

    return f"{line1}\n{line2}\n" if line2 else f"{line1}\n"


def assemble_doc_text(
    paragraphs: List[ParagraphNode],
    tree: DocTree,
    include_prefix: bool = True,
) -> Tuple[str, List[Tuple[str, int, int]]]:
    """Concatenate paragraphs in reading order into a single string.

    Returns:
        (doc_text, spans) where `spans` is a list of (para_id, char_start, char_end)
        indicating the character range of each paragraph's *own text* (excluding any
        prefix) within the assembled doc_text.
    """
    parts: List[str] = []
    spans: List[Tuple[str, int, int]] = []
    offset = 0
    sep = "\n\n"

    for p in paragraphs:
        prefix = build_context_prefix(p, tree) if include_prefix else ""
        if prefix:
            parts.append(prefix)
            offset += len(prefix)
        text = p.text
        start = offset
        parts.append(text)
        offset += len(text)
        end = offset
        spans.append((p.para_id, start, end))
        parts.append(sep)
        offset += len(sep)

    return "".join(parts), spans


# -------------------------- model loading --------------------------

_model_cache: Dict[str, Any] = {}


def _load_embed_model(model_name: Optional[str] = None):
    """Lazy-load a sentence-transformers model. Cached across calls."""
    from ..settings import EMBED_MODEL_V2
    name = model_name or EMBED_MODEL_V2
    if name in _model_cache:
        return _model_cache[name]
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(name, trust_remote_code=True)
    _model_cache[name] = model
    return model


# -------------------------- span mapping --------------------------

def _char_spans_to_token_spans(
    tokenizer,
    doc_text: str,
    char_spans: List[Tuple[str, int, int]],
) -> List[Tuple[str, int, int]]:
    """Convert (para_id, char_start, char_end) spans to token index spans using
    the tokenizer's offset_mapping. Returns (para_id, token_start, token_end)."""
    encoding = tokenizer(
        doc_text,
        return_offsets_mapping=True,
        add_special_tokens=True,
        truncation=False,
    )
    offsets = encoding["offset_mapping"]
    out: List[Tuple[str, int, int]] = []
    for para_id, cs, ce in char_spans:
        tok_start: Optional[int] = None
        tok_end: Optional[int] = None
        for i, (o0, o1) in enumerate(offsets):
            if o0 == 0 and o1 == 0:
                continue  # special token
            if tok_start is None and o1 > cs and o0 < ce:
                tok_start = i
            if o0 < ce:
                tok_end = i + 1
        if tok_start is None or tok_end is None or tok_end <= tok_start:
            out.append((para_id, 0, 0))
            continue
        out.append((para_id, tok_start, tok_end))
    return out


# -------------------------- late chunking (single pass) --------------------------

def late_chunked_embeddings(
    doc_text: str,
    char_spans: List[Tuple[str, int, int]],
    model,
) -> np.ndarray:
    """Encode the full doc and mean-pool token embeddings over each paragraph's span.

    Returns an array of shape [n_paragraphs, dim], L2-normalized.
    """
    import torch

    tokenizer = model.tokenizer
    token_spans = _char_spans_to_token_spans(tokenizer, doc_text, char_spans)

    inputs = tokenizer(
        doc_text,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=False,
    )
    # Drop offset_mapping if the tokenizer included it
    model_inputs = {k: v for k, v in inputs.items() if k != "offset_mapping"}

    # Move to the model's device
    auto_model = model[0].auto_model
    device = next(auto_model.parameters()).device
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

    with torch.no_grad():
        outputs = auto_model(**model_inputs)
    token_embs = outputs.last_hidden_state[0]  # [n_tokens, dim]

    dim = int(token_embs.shape[-1])
    vecs = np.zeros((len(token_spans), dim), dtype=np.float32)
    n_tokens = int(token_embs.shape[0])
    for i, (_pid, s, e) in enumerate(token_spans):
        if e > s and s < n_tokens:
            s_clip = max(0, min(s, n_tokens))
            e_clip = max(0, min(e, n_tokens))
            if e_clip > s_clip:
                vecs[i] = token_embs[s_clip:e_clip].mean(dim=0).cpu().numpy().astype(np.float32)

    # L2 normalize, avoiding div-by-zero
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


# -------------------------- long late chunking dispatcher --------------------------

def vectorize_text_spans(
    doc_text: str,
    char_spans: List[Tuple[str, int, int]],
    model,
) -> np.ndarray:
    """Top-level vectorization: picks single-pass or long late chunking based on length."""
    from ..settings import LATE_CHUNK_MAX_TOKENS, LATE_CHUNK_OVERLAP

    tokenizer = model.tokenizer
    enc = tokenizer(doc_text, add_special_tokens=True, truncation=False)
    n_tokens = len(enc["input_ids"])

    if n_tokens <= LATE_CHUNK_MAX_TOKENS:
        return late_chunked_embeddings(doc_text, char_spans, model)

    return _long_late_chunking(
        doc_text=doc_text,
        char_spans=char_spans,
        model=model,
        max_tokens=LATE_CHUNK_MAX_TOKENS,
        overlap=LATE_CHUNK_OVERLAP,
    )


def _long_late_chunking(
    doc_text: str,
    char_spans: List[Tuple[str, int, int]],
    model,
    max_tokens: int,
    overlap: int,
) -> np.ndarray:
    """Split doc_text into overlapping macro chunks on token boundaries,
    encode each with late chunking, and merge the per-paragraph vectors.

    Each paragraph is attributed to the macro chunk that covers the largest
    fraction of its character span.
    """
    tokenizer = model.tokenizer
    enc = tokenizer(doc_text, return_offsets_mapping=True, add_special_tokens=False, truncation=False)
    input_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    n = len(input_ids)

    dim = int(model[0].auto_model.config.hidden_size)
    vecs = np.zeros((len(char_spans), dim), dtype=np.float32)
    best_coverage = [0] * len(char_spans)
    para_index = {para_id: idx for idx, (para_id, _, _) in enumerate(char_spans)}

    stride = max(1, max_tokens - overlap)
    start = 0
    while start < n:
        end = min(start + max_tokens, n)
        macro_offset_start = offsets[start][0] if start < n else 0
        macro_offset_end = offsets[end - 1][1] if end > 0 else 0
        macro_text = doc_text[macro_offset_start:macro_offset_end]

        relevant: List[Tuple[str, int, int]] = []
        for para_id, cs, ce in char_spans:
            rel_cs = max(cs, macro_offset_start) - macro_offset_start
            rel_ce = min(ce, macro_offset_end) - macro_offset_start
            if rel_ce > rel_cs:
                relevant.append((para_id, rel_cs, rel_ce))

        if relevant:
            macro_vecs = late_chunked_embeddings(macro_text, relevant, model)
            for (para_id, rel_cs, rel_ce), v in zip(relevant, macro_vecs):
                coverage = rel_ce - rel_cs
                idx = para_index[para_id]
                if coverage > best_coverage[idx]:
                    vecs[idx] = v
                    best_coverage[idx] = coverage

        if end >= n:
            break
        start += stride

    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


# -------------------------- tree → embeddings orchestrator --------------------------

def vectorize_tree(doc_id: str, model_name: Optional[str] = None) -> dict:
    """Embed all paragraphs in the DocTree for doc_id and write sidecars.

    Reads: .tree.json + .paragraphs.jsonl
    Writes: .embeddings_v2.npy + .embeddings_v2_meta.json
    """
    from .store import paths, write_json
    from .tree_store import read_tree, read_paragraphs

    p = paths(doc_id)
    tree = read_tree(p["tree"])
    paragraphs = read_paragraphs(p["paragraphs"])
    if not paragraphs:
        return {"doc_id": doc_id, "n_embedded": 0, "dim": None, "model": model_name}

    # Filter out placeholder/opaque content that shouldn't be embedded
    embeddable = [q for q in paragraphs if q.text and not q.text.startswith("[[") and len(q.text) >= 20]
    if not embeddable:
        return {"doc_id": doc_id, "n_embedded": 0, "dim": None, "model": model_name}

    doc_text, char_spans = assemble_doc_text(embeddable, tree, include_prefix=True)
    model = _load_embed_model(model_name)
    vecs = vectorize_text_spans(doc_text, char_spans, model)

    np.save(p["embeddings_v2"], vecs)

    from ..settings import EMBED_MODEL_V2, LATE_CHUNK_MAX_TOKENS, LATE_CHUNK_OVERLAP
    meta = {
        "model": model_name or EMBED_MODEL_V2,
        "dim": int(vecs.shape[1]),
        "para_ids": [q.para_id for q in embeddable],
        "prefix_strategy": "metadata+section_path+label",
        "late_chunking": True,
        "macro_chunk_tokens": LATE_CHUNK_MAX_TOKENS,
        "macro_chunk_overlap": LATE_CHUNK_OVERLAP,
    }
    write_json(p["embeddings_v2_meta"], meta)

    return {
        "doc_id": doc_id,
        "n_embedded": len(embeddable),
        "dim": int(vecs.shape[1]),
        "model": meta["model"],
    }
