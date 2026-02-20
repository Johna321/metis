from __future__ import annotations
import numpy as np
from sentence_transformers import SentenceTransformer
from ..settings import EMBED_MODEL


class MetisDenseRetriever:
    """Wraps a SentenceTransformer model in the interface BEIR's
    DenseRetrievalExactSearch expects: encode_queries and encode_corpus."""

    def __init__(self, model_name: str | None = None, batch_size: int = 64):
        self.model_name = model_name or EMBED_MODEL
        self.model = SentenceTransformer(self.model_name)
        self.batch_size = batch_size

    def encode_queries(self, queries: list[str], batch_size: int = 0, **kwargs) -> np.ndarray:
        bs = batch_size or self.batch_size
        return self.model.encode(
            queries, batch_size=bs, normalize_embeddings=True,
            show_progress_bar=kwargs.get("show_progress_bar", False),
        )

    def encode_corpus(self, corpus: list[dict], batch_size: int = 0, **kwargs) -> np.ndarray:
        bs = batch_size or self.batch_size
        sentences = [
            (doc.get("title", "") + " " + doc.get("text", "")).strip()
            for doc in corpus
        ]
        return self.model.encode(
            sentences, batch_size=bs, normalize_embeddings=True,
            show_progress_bar=kwargs.get("show_progress_bar", False),
        )
