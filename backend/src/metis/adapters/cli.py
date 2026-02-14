import logging
from enum import Enum

import typer
import pymupdf
from rich import print
from pathlib import Path
from ..core.ingest import ingest_pdf_bytes, ingest_pdf_bytes_layout
from ..core.retrieve import retrieve
from ..core.store import paths, read_spans_jsonl

app = typer.Typer(no_args_is_help=True)

_KIND_COLORS = {
    "text":    (1, 0, 0),      # red
    "table":   (1, 0.6, 0),    # orange
    "picture": (0, 0, 1),      # blue
}
_KIND_DEFAULT_COLOR = (0.5, 0, 0.5)  # purple for unknown kinds


class Engine(str, Enum):
    blocks = "blocks"
    layout = "layout"


@app.command()
def ingest(
    pdf: Path,
    engine: Engine = typer.Option(Engine.blocks, "--engine", help="Ingestion engine"),
    extract_words: bool = typer.Option(False, "--extract-words/--no-extract-words", help="Extract word-level bboxes (layout only)"),
    write_images: bool = typer.Option(False, "--write-images/--no-write-images", help="Materialize images (layout only)"),
    dpi: int = typer.Option(200, "--dpi", help="Image DPI (layout only)"),
):
    # Enable info logging so layout engine counts are visible
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if engine == Engine.layout:
        meta = ingest_pdf_bytes_layout(
            pdf.read_bytes(),
            extract_words=extract_words,
            write_images=write_images,
            dpi=dpi,
        )
    else:
        meta = ingest_pdf_bytes(pdf.read_bytes())
    print(meta)


@app.command()
def retrieve_evidence(doc_id: str, page: int, text: str):
    ev = retrieve(doc_id=doc_id, page=page, selected_text=text)
    print([e.__dict__ for e in ev])


@app.command()
def debug_page(
        doc_id: str,
        output: Path,
        start: int = typer.Option(None, "--start", "-s"),
        end: int = typer.Option(None, "--end", "-e"),
        raw: bool = typer.Option(False, "--raw"),
        show_blocks: bool = typer.Option(False, "--blocks"),
        show_draws: bool = typer.Option(False, "--draws"),
        show_layout: bool = typer.Option(False, "--layout-boxes", help="Draw layout span bboxes colored by kind"),
        show_words: bool = typer.Option(False, "--words", help="Draw word-level bboxes (gray)"),
    ):
    p = paths(doc_id)
    doc = pymupdf.open(p["pdf"])

    start = start if start is not None else 0
    end = end if end is not None else len(doc) - 1

    if raw:
        # re-extract direct from PDF
        blocks_by_page = {
            i: [{"bbox_pdf": b[:4], "span_id": f"raw_{j}"}
                for j, b in enumerate(doc[i].get_text("blocks")) if b[6] == 0]
            for i in range(start, end + 1)
        }
    else:
        # use stored spans
        all_spans = read_spans_jsonl(p["spans"])
        blocks_by_page = {}
        for s in all_spans:
            if start <= s.page <= end:
                blocks_by_page.setdefault(s.page, []).append(s)

    draws_by_page = {
        i: doc[i].cluster_drawings()
        for i in range(start, end + 1)
    } if show_draws else {}

    # Load words if requested
    words_by_page = {}
    if show_words:
        words_path = p["page_md"].with_suffix(".words.json")
        if words_path.exists():
            import orjson
            words_by_page = orjson.loads(words_path.read_bytes())
            # Keys are strings from JSON; convert to int
            words_by_page = {int(k): v for k, v in words_by_page.items()}
        else:
            # Fall back: re-extract words on the fly for the requested pages
            print("[yellow]No stored words found; extracting on-the-fly...[/yellow]")
            try:
                from ..core.ingest import ensure_pymupdf4llm
                _pymupdf4llm = ensure_pymupdf4llm()
                chunks = _pymupdf4llm.to_markdown(
                    doc,
                    page_chunks=True,
                    extract_words=True,
                )
                for pi, chunk in enumerate(chunks):
                    if start <= pi <= end and "words" in chunk:
                        words_by_page[pi] = chunk["words"]
            except (ImportError, RuntimeError):
                print("[red]pymupdf4llm not installed; cannot extract words[/red]")

    for page_num in range(start, end + 1):
        page = doc[page_num]

        # Original block outlines (red)
        if show_blocks:
            for b in blocks_by_page.get(page_num, []):
                bbox = b.bbox_pdf if hasattr(b, "bbox_pdf") else b["bbox_pdf"]
                span_id = b.span_id if hasattr(b, "span_id") else b["span_id"]
                rect = pymupdf.Rect(bbox)
                page.draw_rect(rect, color=(1, 0, 0), width=0.5)
                page.insert_text((rect.x0, rect.y0 - 2), span_id, fontsize=6, color=(1, 0, 0))

        # Layout-boxes: colored by kind
        if show_layout:
            for b in blocks_by_page.get(page_num, []):
                bbox = b.bbox_pdf if hasattr(b, "bbox_pdf") else b.get("bbox_pdf")
                span_id = b.span_id if hasattr(b, "span_id") else b.get("span_id", "")
                kind = b.kind if hasattr(b, "kind") else b.get("kind")
                if bbox is None:
                    continue
                color = _KIND_COLORS.get(kind, _KIND_DEFAULT_COLOR)
                rect = pymupdf.Rect(bbox)
                page.draw_rect(rect, color=color, width=0.8)
                label = f"{span_id} [{kind}]" if kind else span_id
                page.insert_text((rect.x0, rect.y0 - 2), label, fontsize=5, color=color)

        # Drawing clusters (green)
        if show_draws:
            for i, rect in enumerate(draws_by_page.get(page_num, [])):
                page.draw_rect(rect, color=(0, 1, 0), width=0.5)
                page.insert_text((rect.x1 - 30, rect.y0 - 2), f"draw_{i}", fontsize=6, color=(0, 1, 0))

        # Word-level bboxes (thin gray)
        if show_words and page_num in words_by_page:
            for word in words_by_page[page_num]:
                # word may be a dict with "bbox" or a list [x0,y0,x1,y1,text,...]
                if isinstance(word, dict):
                    wb = word.get("bbox")
                elif isinstance(word, (list, tuple)) and len(word) >= 4:
                    wb = word[:4]
                else:
                    continue
                if wb is None:
                    continue
                rect = pymupdf.Rect(wb)
                page.draw_rect(rect, color=(0.5, 0.5, 0.5), width=0.3)

    out_doc = pymupdf.open()
    out_doc.insert_pdf(doc, from_page=start, to_page=end)
    out_doc.save(output)
    print(f"[green]Saved debug PDF → {output}[/green]")


def main():
    app()
