# Metis

Minimal PDF viewer for research papers with integrated LLM tooling that provides contextual explanations.

## Project Structure

- `backend/` - Python backend (primary implementation)
- `metis-frontend/` - React-based frontend

## Backend Usage (Python)

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

Additional options:
- `--engine [blocks|layout]` - Choose ingestion engine (default: blocks)
- `--extract-words` - Extract word-level bounding boxes (layout engine only)
- `--write-images` - Materialize images during ingestion (layout engine only)
- `--dpi <number>` - Set image DPI for extraction (default: 200, layout engine only)

**Fuzzy search for text**

```bash
uv run metis retrieve-evidence <doc_id> <page> "<search_text>"
```

Performs a fuzzy search on the specified page of an ingested document. Returns matching text spans ranked by similarity score, along with neighboring context.

**Debug page rendering**

```bash
uv run metis debug-page <doc_id> <output_path.pdf>
```

Creates a debug PDF with visual annotations to inspect ingested document structure.

Options:
- `--start <page>` / `-s <page>` - Start page (default: 0)
- `--end <page>` / `-e <page>` - End page (default: last page)
- `--raw` - Extract directly from PDF instead of using stored spans
- `--blocks` - Draw text block outlines (red)
- `--draws` - Draw drawing clusters (green)
- `--layout-boxes` - Draw layout span bounding boxes colored by kind
- `--words` - Draw word-level bounding boxes (gray)

### Web Server (API)

The backend also provides a FastAPI web server that exposes the same functionality as the CLI over HTTP. This is used by the Tauri frontend.

**Start the server:**

```bash
cd backend
uv run metis-web
```

The server runs on `http://localhost:8000`. CORS is configured for the Tauri dev server (`localhost:1420`).

#### API Endpoints

**`POST /ingest`** — Upload and ingest a PDF

Accepts a multipart file upload with optional query parameters.

```bash
curl -X POST http://localhost:8000/ingest -F "file=@data/test.pdf"

# With options:
curl -X POST "http://localhost:8000/ingest?engine=layout&extract_words=true&dpi=300" \
  -F "file=@data/test.pdf"
```

Query parameters: `engine` (blocks|layout), `extract_words`, `write_images`, `dpi`.

Returns: `{ "doc_id": "sha256:...", "n_pages": N, "n_spans": N, "ingest": {...} }`

**`POST /retrieve`** — Retrieve evidence for selected text

```bash
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "sha256:...", "page": 0, "selected_text": "some text from the PDF"}'
```

Returns: `[{ "span_id": "...", "page": 0, "bbox_norm": [x0,y0,x1,y1], "text": "...", "score": 95.0 }, ...]`

**`GET /documents/{doc_id}`** — Get document metadata

```bash
curl http://localhost:8000/documents/sha256:abc123...
```

Returns the contents of the document's `doc.json`.

**`GET /documents/{doc_id}/pdf`** — Serve the original PDF

```bash
curl -o output.pdf http://localhost:8000/documents/sha256:abc123.../pdf
```

Returns the PDF file with `Content-Type: application/pdf`.

### Testing the Backend

Since the backend doesn't have automated tests yet, manual testing is done through the CLI commands. Here's a typical testing workflow:

1. **Place a test PDF in the data directory:**
   ```bash
   cp /path/to/test.pdf backend/data/
   ```

2. **Test ingestion with blocks engine:**
   ```bash
   cd backend
   uv run metis ingest data/test.pdf
   ```

   This should output document metadata including a `doc_id` (usually derived from the filename).

3. **Test ingestion with layout engine:**
   ```bash
   uv run metis ingest --engine layout --extract-words data/test.pdf
   ```

4. **Test retrieval:**
   ```bash
   uv run metis retrieve-evidence <doc_id> 0 "some text from the first page"
   ```

   Replace `<doc_id>` with the ID from step 2, and use actual text that appears in your PDF.

5. **Generate debug visualization:**
   ```bash
   uv run metis debug-page <doc_id> debug_output.pdf --blocks --layout-boxes --words
   ```

   Open `debug_output.pdf` to visually inspect the extracted spans and bounding boxes.

**Expected behavior:**
- Ingestion should complete without errors and output valid JSON metadata
- Retrieval should return relevant text spans with similarity scores
- Debug PDFs should show colored annotations corresponding to detected text blocks
