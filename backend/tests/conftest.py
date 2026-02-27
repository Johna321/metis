"""Shared pytest fixtures for Metis backend tests."""
from __future__ import annotations
import pytest
import pymupdf
from fastapi.testclient import TestClient
from metis.adapters.web import app


@pytest.fixture(scope="session")
def pdf_bytes() -> bytes:
    """Minimal 1-page PDF with 4 known sentences (each > 20 chars for MIN_CHARS filter)."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    sentences = [
        "The attention mechanism allows the model to focus on relevant tokens.",
        "Transformers use self-attention for sequence to sequence modeling tasks.",
        "Neural networks learn distributed representations from raw data.",
        "Gradient descent optimizes the loss function across training iterations.",
    ]
    y = 100
    for sentence in sentences:
        page.insert_text((50, y), sentence, fontsize=11)
        y += 40
    return doc.tobytes()


@pytest.fixture()
def client(tmp_path, monkeypatch) -> TestClient:
    """TestClient with DATA_DIR patched to tmp_path (isolated per test)."""
    monkeypatch.setattr("metis.core.store.DATA_DIR", tmp_path)
    return TestClient(app)


@pytest.fixture()
def ingested_doc(client: TestClient, pdf_bytes: bytes) -> str:
    """Ingest the test PDF, return doc_id."""
    resp = client.post("/ingest", files={"file": ("test.pdf", pdf_bytes, "application/pdf")})
    assert resp.status_code == 200, f"Ingest failed: {resp.text}"
    return resp.json()["doc_id"]


@pytest.fixture()
def vectorized_doc(client: TestClient, ingested_doc: str) -> str:
    """Ingest + vectorize test doc, return doc_id."""
    resp = client.post("/vectorize", json={"doc_id": ingested_doc})
    assert resp.status_code == 200, f"Vectorize failed: {resp.text}"
    return ingested_doc
