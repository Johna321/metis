import typer
from rich import print
from pathlib import Path
from ..core.ingest import ingest_pdf_bytes
from ..core.retrieve import retrieve

app = typer.Typer(no_args_is_help=True)

@app.command()
def ingest(pdf: Path):
    meta = ingest_pdf_bytes(pdf.read_bytes())
    print(meta)

@app.command()
def retrieve_evidence(doc_id: str, page: int, text: str):
    ev = retrieve(doc_id=doc_id, page=page, selected_text=text)
    print([e.__dict__ for e in ev])

def main():
    app()

