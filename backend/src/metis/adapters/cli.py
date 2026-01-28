import typer
import pymupdf
from rich import print
from pathlib import Path
from ..core.ingest import ingest_pdf_bytes
from ..core.retrieve import retrieve
from ..core.store import paths, read_spans_jsonl

app = typer.Typer(no_args_is_help=True)

@app.command()
def ingest(pdf: Path):
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
        show_draws: bool = typer.Option(False, "--draws")
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
        # use store doc_id
        all_spans = read_spans_jsonl(p["spans"])
        blocks_by_page = {}
        for s in all_spans:
            if start <= s.page <= end:
                blocks_by_page.setdefault(s.page, []).append(s)

    draws_by_page = {
        i: doc[i].cluster_drawings()
        for i in range(start, end + 1)
    } if show_draws else {}

    for page_num in range(start, end + 1):
        page = doc[page_num]
        if show_blocks:
            for b in blocks_by_page.get(page_num, []):
                bbox = b.bbox_pdf if hasattr(b, "bbox_pdf") else b["bbox_pdf"]
                span_id = b.span_id if hasattr(b, "span_id") else b["span_id"]
                rect = pymupdf.Rect(bbox)
                page.draw_rect(rect, color=(1, 0, 0), width=0.5)
                page.insert_text((rect.x0, rect.y0 - 2), span_id, fontsize=6, color=(1, 0, 0))
        if show_draws:
            for i, rect in enumerate(draws_by_page.get(page_num, [])):
                page.draw_rect(rect, color=(0, 1, 0), width=0.5)
                page.insert_text((rect.x1 - 30, rect.y0 - 2), f"draw_{i}", fontsize=6, color=(0, 1, 0))

    out_doc = pymupdf.open()
    out_doc.insert_pdf(doc, from_page=start, to_page=end)
    out_doc.save(output)

def main():
    app()

