"""Integration tests for all FastAPI routes in metis.adapters.web."""
from __future__ import annotations

from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# POST /ingest
# ---------------------------------------------------------------------------

class TestIngest:
    def test_ingest_returns_200_with_valid_pdf(self, client: TestClient, pdf_bytes: bytes):
        resp = client.post("/ingest", files={"file": ("test.pdf", pdf_bytes, "application/pdf")})
        assert resp.status_code == 200

    def test_ingest_response_schema(self, client: TestClient, pdf_bytes: bytes):
        resp = client.post("/ingest", files={"file": ("test.pdf", pdf_bytes, "application/pdf")})
        data = resp.json()
        assert {"doc_id", "n_pages", "n_spans", "ingest"} == set(data.keys())

    def test_ingest_doc_id_is_sha256_prefixed(self, client: TestClient, pdf_bytes: bytes):
        resp = client.post("/ingest", files={"file": ("test.pdf", pdf_bytes, "application/pdf")})
        assert resp.json()["doc_id"].startswith("sha256:")

    def test_ingest_n_pages_is_one(self, client: TestClient, pdf_bytes: bytes):
        resp = client.post("/ingest", files={"file": ("test.pdf", pdf_bytes, "application/pdf")})
        assert resp.json()["n_pages"] == 1

    def test_ingest_n_spans_is_nonzero(self, client: TestClient, pdf_bytes: bytes):
        resp = client.post("/ingest", files={"file": ("test.pdf", pdf_bytes, "application/pdf")})
        assert resp.json()["n_spans"] > 0

    def test_ingest_default_engine_is_blocks(self, client: TestClient, pdf_bytes: bytes):
        resp = client.post("/ingest", files={"file": ("test.pdf", pdf_bytes, "application/pdf")})
        assert resp.json()["ingest"]["engine"] == "pymupdf"

    def test_ingest_layout_engine(self, client: TestClient, pdf_bytes: bytes):
        resp = client.post(
            "/ingest?engine=layout",
            files={"file": ("test.pdf", pdf_bytes, "application/pdf")},
        )
        assert resp.status_code == 200
        assert resp.json()["ingest"]["engine"] == "pymupdf4llm"

    def test_ingest_is_deterministic(self, client: TestClient, pdf_bytes: bytes):
        r1 = client.post("/ingest", files={"file": ("test.pdf", pdf_bytes, "application/pdf")})
        r2 = client.post("/ingest", files={"file": ("test.pdf", pdf_bytes, "application/pdf")})
        assert r1.json()["doc_id"] == r2.json()["doc_id"]


# ---------------------------------------------------------------------------
# POST /retrieve
# ---------------------------------------------------------------------------

class TestRetrieve:
    def test_retrieve_returns_200_for_known_text(self, client: TestClient, ingested_doc: str):
        resp = client.post(
            "/retrieve",
            json={"doc_id": ingested_doc, "page": 0, "selected_text": "attention mechanism"},
        )
        assert resp.status_code == 200

    def test_retrieve_returns_list(self, client: TestClient, ingested_doc: str):
        resp = client.post(
            "/retrieve",
            json={"doc_id": ingested_doc, "page": 0, "selected_text": "attention mechanism"},
        )
        assert isinstance(resp.json(), list)

    def test_retrieve_result_schema(self, client: TestClient, ingested_doc: str):
        resp = client.post(
            "/retrieve",
            json={"doc_id": ingested_doc, "page": 0, "selected_text": "attention mechanism"},
        )
        results = resp.json()
        assert len(results) > 0
        item = results[0]
        assert {"span_id", "page", "bbox_norm", "text", "score"} <= set(item.keys())

    def test_retrieve_top_result_contains_query_text(self, client: TestClient, ingested_doc: str):
        resp = client.post(
            "/retrieve",
            json={"doc_id": ingested_doc, "page": 0, "selected_text": "attention mechanism"},
        )
        results = resp.json()
        assert len(results) > 0
        assert "attention" in results[0]["text"].lower()

    def test_retrieve_bbox_norm_is_four_floats(self, client: TestClient, ingested_doc: str):
        resp = client.post(
            "/retrieve",
            json={"doc_id": ingested_doc, "page": 0, "selected_text": "attention mechanism"},
        )
        results = resp.json()
        assert len(results) > 0
        bbox = results[0]["bbox_norm"]
        assert len(bbox) == 4
        assert all(isinstance(v, float) for v in bbox)

    def test_retrieve_404_for_missing_doc(self, client: TestClient):
        resp = client.post(
            "/retrieve",
            json={"doc_id": "sha256:doesnotexist", "page": 0, "selected_text": "hello"},
        )
        assert resp.status_code == 404

    def test_retrieve_empty_for_nonexistent_page(self, client: TestClient, ingested_doc: str):
        resp = client.post(
            "/retrieve",
            json={"doc_id": ingested_doc, "page": 99, "selected_text": "attention"},
        )
        assert resp.status_code == 200
        assert resp.json() == []


# ---------------------------------------------------------------------------
# POST /vectorize
# ---------------------------------------------------------------------------

class TestVectorize:
    def test_vectorize_returns_200(self, client: TestClient, ingested_doc: str):
        resp = client.post("/vectorize", json={"doc_id": ingested_doc})
        assert resp.status_code == 200

    def test_vectorize_response_schema(self, client: TestClient, ingested_doc: str):
        resp = client.post("/vectorize", json={"doc_id": ingested_doc})
        data = resp.json()
        assert {"doc_id", "n_embedded", "model"} <= set(data.keys())

    def test_vectorize_doc_id_matches_request(self, client: TestClient, ingested_doc: str):
        resp = client.post("/vectorize", json={"doc_id": ingested_doc})
        assert resp.json()["doc_id"] == ingested_doc

    def test_vectorize_n_embedded_is_nonzero(self, client: TestClient, ingested_doc: str):
        resp = client.post("/vectorize", json={"doc_id": ingested_doc})
        assert resp.json()["n_embedded"] > 0

    def test_vectorize_model_is_string(self, client: TestClient, ingested_doc: str):
        resp = client.post("/vectorize", json={"doc_id": ingested_doc})
        assert isinstance(resp.json()["model"], str)

    def test_vectorize_404_for_missing_doc(self, client: TestClient):
        resp = client.post("/vectorize", json={"doc_id": "sha256:doesnotexist"})
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /retrieve-semantic
# ---------------------------------------------------------------------------

class TestRetrieveSemantic:
    def test_retrieve_semantic_returns_200(self, client: TestClient, vectorized_doc: str):
        resp = client.post(
            "/retrieve-semantic",
            json={"doc_id": vectorized_doc, "query": "attention mechanism"},
        )
        assert resp.status_code == 200

    def test_retrieve_semantic_returns_list(self, client: TestClient, vectorized_doc: str):
        resp = client.post(
            "/retrieve-semantic",
            json={"doc_id": vectorized_doc, "query": "attention mechanism"},
        )
        assert isinstance(resp.json(), list)

    def test_retrieve_semantic_result_schema(self, client: TestClient, vectorized_doc: str):
        resp = client.post(
            "/retrieve-semantic",
            json={"doc_id": vectorized_doc, "query": "attention mechanism"},
        )
        results = resp.json()
        assert len(results) > 0
        item = results[0]
        assert {"span_id", "page", "bbox_norm", "text", "score"} <= set(item.keys())

    def test_retrieve_semantic_scores_are_floats(self, client: TestClient, vectorized_doc: str):
        resp = client.post(
            "/retrieve-semantic",
            json={"doc_id": vectorized_doc, "query": "attention mechanism"},
        )
        results = resp.json()
        assert len(results) > 0
        assert all(isinstance(r["score"], float) for r in results)

    def test_retrieve_semantic_top_k_one(self, client: TestClient, vectorized_doc: str):
        resp = client.post(
            "/retrieve-semantic",
            json={"doc_id": vectorized_doc, "query": "attention", "top_k": 1},
        )
        assert resp.status_code == 200
        assert len(resp.json()) <= 1

    def test_retrieve_semantic_top_k_two(self, client: TestClient, vectorized_doc: str):
        resp = client.post(
            "/retrieve-semantic",
            json={"doc_id": vectorized_doc, "query": "attention", "top_k": 2},
        )
        assert resp.status_code == 200
        assert len(resp.json()) <= 2

    def test_retrieve_semantic_page_filter_returns_matching_page(
        self, client: TestClient, vectorized_doc: str
    ):
        resp = client.post(
            "/retrieve-semantic",
            json={"doc_id": vectorized_doc, "query": "attention", "page": 0},
        )
        assert resp.status_code == 200
        results = resp.json()
        assert all(r["page"] == 0 for r in results)

    def test_retrieve_semantic_page_filter_nonexistent_page_empty(
        self, client: TestClient, vectorized_doc: str
    ):
        resp = client.post(
            "/retrieve-semantic",
            json={"doc_id": vectorized_doc, "query": "attention", "page": 99},
        )
        assert resp.status_code == 200
        assert resp.json() == []

    def test_retrieve_semantic_404_when_embeddings_missing(
        self, client: TestClient, ingested_doc: str
    ):
        # spans exist but no embeddings
        resp = client.post(
            "/retrieve-semantic",
            json={"doc_id": ingested_doc, "query": "attention mechanism"},
        )
        assert resp.status_code == 404

    def test_retrieve_semantic_404_for_missing_doc(self, client: TestClient):
        resp = client.post(
            "/retrieve-semantic",
            json={"doc_id": "sha256:doesnotexist", "query": "attention"},
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /documents/{doc_id}
# ---------------------------------------------------------------------------

class TestGetDocument:
    def test_get_document_returns_200(self, client: TestClient, ingested_doc: str):
        resp = client.get(f"/documents/{ingested_doc}")
        assert resp.status_code == 200

    def test_get_document_response_schema(self, client: TestClient, ingested_doc: str):
        resp = client.get(f"/documents/{ingested_doc}")
        data = resp.json()
        assert {"doc_id", "n_pages", "n_spans", "ingest"} <= set(data.keys())

    def test_get_document_doc_id_matches(self, client: TestClient, ingested_doc: str):
        resp = client.get(f"/documents/{ingested_doc}")
        assert resp.json()["doc_id"] == ingested_doc

    def test_get_document_n_pages_is_positive(self, client: TestClient, ingested_doc: str):
        resp = client.get(f"/documents/{ingested_doc}")
        assert resp.json()["n_pages"] > 0

    def test_get_document_404_for_missing_doc(self, client: TestClient):
        resp = client.get("/documents/sha256:doesnotexist")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /documents/{doc_id}/pdf
# ---------------------------------------------------------------------------

class TestGetDocumentPdf:
    def test_get_document_pdf_returns_200(self, client: TestClient, ingested_doc: str):
        resp = client.get(f"/documents/{ingested_doc}/pdf")
        assert resp.status_code == 200

    def test_get_document_pdf_content_type(self, client: TestClient, ingested_doc: str):
        resp = client.get(f"/documents/{ingested_doc}/pdf")
        assert "application/pdf" in resp.headers["content-type"]

    def test_get_document_pdf_magic_bytes(self, client: TestClient, ingested_doc: str):
        resp = client.get(f"/documents/{ingested_doc}/pdf")
        assert resp.content[:5] == b"%PDF-"

    def test_get_document_pdf_is_nonempty(self, client: TestClient, ingested_doc: str):
        resp = client.get(f"/documents/{ingested_doc}/pdf")
        assert len(resp.content) > 100

    def test_get_document_pdf_404_for_missing_doc(self, client: TestClient):
        resp = client.get("/documents/sha256:doesnotexist/pdf")
        assert resp.status_code == 404
