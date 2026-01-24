# Metis

Minimal PDF viewer for research papers with integrated LLM tooling that provides contextual explanations.

## Backend Usage

Metis is currently in the prototype stage and is not connected to the frontend. As such, all commands should be run from the `backend/` directory.
The backend is a Python project managed with [uv](https://docs.astral.sh/uv/).

### Setup

```bash
cd backend
uv sync
```

Installs all dependencies from `pyproject.toml` into a local `.venv`.

### Commands

**Ingest a PDF**

```bash
uv run metis ingest data/<filename>.pdf
```

Parses a PDF file and extracts text spans with positional metadata. Outputs document metadata including a `doc_id` for use with retrieval.

**Fuzzy search for text**

```bash
uv run metis retrieve-evidence <doc_id> <page> "<search_text>"
```

Performs a fuzzy search on the specified page of an ingested document. Returns matching text spans ranked by similarity score, along with neighboring context.
