import json
import logging
from enum import Enum

import typer
import pymupdf
from rich import print
from pathlib import Path
from ..core.ingest import ingest_pdf_bytes, ingest_pdf_bytes_layout
from ..core.retrieve import retrieve
from ..core.store import paths, read_spans_jsonl
from ..core.vectorize import vectorize_spans, retrieve_semantic
from ..core.agent import run_agent
from ..core.llm import AnthropicModel, OpenAIModel, StreamEvent
from ..core.tools import ToolRegistry, make_rag_retrieve_tool, make_web_search_tool
from ..core.prompts import SYSTEM_PROMPT
from ..settings import (
    LLM_PROVIDER, LLM_MODEL, LLM_API_KEY, TAVILY_API_KEY,
    AGENT_MAX_ITER, AGENT_TEMPERATURE,
)
from ..benchmark.runner import run_retrieval_benchmark, AVAILABLE_DATASETS
from ..benchmark.ingestion import ingestion_metrics

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


@app.command()
def vectorize(doc_id: str):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    meta = vectorize_spans(doc_id)
    print(meta)


@app.command("retrieve-semantic")
def retrieve_semantic_cmd(
    doc_id: str,
    query: str,
    page: int = typer.Option(None, "--page", "-p", help="Filter to specific page"),
    top_k: int = typer.Option(None, "--top-k", "-k", help="Max results"),
):
    kwargs = {}
    if page is not None:
        kwargs["page"] = page
    if top_k is not None:
        kwargs["top_k"] = top_k
    ev = retrieve_semantic(doc_id=doc_id, query=query, **kwargs)
    print([e.__dict__ for e in ev])

@app.command()
def chat(
    doc_id: str,
    provider: str = typer.Option(None, "--provider", "-p", help="LLM provider: anthropic or openai"),
    model: str = typer.Option(None, "--model", "-m", help="Model ID"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show tool arguments and retrieved chunks"),
):
    """Interactive Q&A chat with a PDF file"""
    from rich.console import Console
    import os

    console = Console()

    # Resolve provider and model
    prov = provider or LLM_PROVIDER
    mod = model or LLM_MODEL
    api_key = LLM_API_KEY

    # Fall back to standard env vars for API key
    if not api_key:
        if prov == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY", "")
        elif prov == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "")

    if not api_key:
        console.print("[red]No API key found. Set METIS_LLM_API_KEY or ANTHROPIC_API_KEY / OPENAI_API_KEY.[/red]")
        raise typer.Exit(1)

    # Validate doc exists and has embeddings
    p = paths(doc_id)
    if not p["spans"].exists():
        console.print(f"[red]Document not found: {doc_id}[/red]")
        console.print("Run: metis ingest <pdf> first.")
        raise typer.Exit(1)

    if not p["embeddings"].exists():
        console.print(f"[yellow]No embeddings found for this document.[/yellow]")
        console.print(f"Run: metis vectorize {doc_id}")
        raise typer.Exit(1)

    # Load doc metadata
    if p["doc"].exists():
        import orjson
        doc_meta = orjson.loads(p["doc"].read_bytes())
        console.print(f"[bold]Paper:[/bold] {doc_meta.get('title', doc_id)}")
        console.print(f"[dim]Pages: {doc_meta.get('page_count', '?')}  |  Provider: {prov}  |  Model: {mod}[/dim]")
    else:
        console.print(f"[bold]Document:[/bold] {doc_id}")

    console.print("[dim]Type 'exit' or press Ctrl+C to quit.[/dim]\n")

    # Build model
    if prov == "anthropic":
        llm = AnthropicModel(api_key=api_key, model=mod, temperature=AGENT_TEMPERATURE)
    elif prov == "openai":
        llm = OpenAIModel(api_key=api_key, model=mod, temperature=AGENT_TEMPERATURE)
    else:
        console.print(f"[red]Unknown provider: {prov}. Use 'anthropic' or 'openai'.[/red]")
        raise typer.Exit(1)

    # Build tools
    registry = ToolRegistry()
    rag_def, rag_fn = make_rag_retrieve_tool(doc_id)
    registry.register(rag_def.name, rag_def.description, rag_def.parameters, rag_fn)

    tavily_key = TAVILY_API_KEY or os.getenv("TAVILY_API_KEY", "")
    if tavily_key:
        ws_def, ws_fn = make_web_search_tool(tavily_key)
        registry.register(ws_def.name, ws_def.description, ws_def.parameters, ws_fn)
    else:
        console.print("[yellow]No Tavily API key: web search disabled.[/yellow]")

    # Stream callback for terminal output
    def on_stream(event: StreamEvent) -> None:
        if event.kind == "text_delta" and event.text:
            console.print(event.text, end="", highlight=False)
        elif event.kind == "tool_call_start" and event.text:
            console.print(f"\n[dim italic]  ↳ Calling {event.text}...[/dim italic]", end="")
        elif event.kind == "tool_call_done":
            console.print(" [dim]done[/dim]")
            if verbose and event.tool_call:
                console.print(f"    [dim cyan]args: {json.dumps(event.tool_call.arguments)}[/dim cyan]")

    # Debug callback for tool results
    def on_tool_result(tool_name: str, arguments: dict, result_str: str) -> None:
        if not verbose:
            return
        parsed = json.loads(result_str)
        if tool_name == "rag_retrieve" and isinstance(parsed, list):
            for i, chunk in enumerate(parsed):
                score = chunk.get("score", 0)
                page = chunk.get("page", "?")
                text = chunk.get("text", "")
                # Truncate long text for display
                display_text = text[:120] + "..." if len(text) > 120 else text
                console.print(f"    [dim yellow]#{i+1} (p{page}, score={score:.3f}): {display_text}[/dim yellow]")
        elif tool_name == "web_search" and isinstance(parsed, list):
            for i, r in enumerate(parsed):
                title = r.get("title", "")
                url = r.get("url", "")
                console.print(f"    [dim yellow]#{i+1}: {title} — {url}[/dim yellow]")
        else:
            # Generic: show truncated JSON
            display = result_str[:200] + "..." if len(result_str) > 200 else result_str
            console.print(f"    [dim yellow]{display}[/dim yellow]")

    # Interactive loop
    try:
        while True:
            try:
                user_input = console.input("[bold green]You > [/bold green]")
            except EOFError:
                break

            if user_input.strip().lower() in ("exit", "quit"):
                break
            if not user_input.strip():
                continue

            result = run_agent(
                model=llm,
                user_query=user_input,
                tools=registry,
                system_prompt=SYSTEM_PROMPT,
                max_iterations=AGENT_MAX_ITER,
                on_stream=on_stream,
                on_tool_result=on_tool_result,
            )
            console.print()  # newline after streamed response
            console.print()  # blank line before next prompt

    except KeyboardInterrupt:
        console.print("\n[dim]Goodbye.[/dim]")

bench_app = typer.Typer(no_args_is_help=True, help="Run benchmarks")
app.add_typer(bench_app, name="benchmark")


@bench_app.command()
def retrieval(
    dataset: str = typer.Option("scifact", "--dataset", "-d", help=f"Dataset name: {', '.join(AVAILABLE_DATASETS)} or 'all'"),
    model: str = typer.Option(None, "--model", "-m", help="Embedding model name (default: from settings)"),
):
    """Run retrieval benchmark on a BEIR dataset."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    datasets = list(AVAILABLE_DATASETS) if dataset == "all" else [dataset]
    for ds in datasets:
        print(f"[bold]Running retrieval benchmark: {ds}[/bold]")
        result = run_retrieval_benchmark(dataset_name=ds, model_name=model)
        print(f"  nDCG@10:     {result.ndcg.get('NDCG@10', 0):.4f}")
        print(f"  MAP@10:      {result.map_score.get('MAP@10', 0):.4f}")
        print(f"  Recall@100:  {result.recall.get('Recall@100', 0):.4f}")
        print(f"  Time:        {result.retrieval_time_s:.1f}s")
        print()


@bench_app.command()
def ingestion(
    annotations_dir: Path = typer.Option("data/benchmark", help="Dir with gold annotation JSON files"),
):
    """Run ingestion quality benchmark against gold annotations."""
    import orjson
    annotations_dir = Path(annotations_dir)
    if not annotations_dir.exists():
        print(f"[red]Annotations dir not found: {annotations_dir}[/red]")
        print("Create gold annotations as JSON files in this directory.")
        raise typer.Exit(1)
    for ann_file in sorted(annotations_dir.glob("*.gold.json")):
        gold_data = orjson.loads(ann_file.read_bytes())
        doc_id = gold_data["doc_id"]
        print(f"[bold]Evaluating: {ann_file.name}[/bold]")
        try:
            spans = read_spans_jsonl(paths(doc_id)["spans"])
        except FileNotFoundError:
            print(f"  [red]Spans not found for {doc_id} — ingest the PDF first[/red]")
            continue
        for page_num, gold_spans in gold_data.get("pages", {}).items():
            predicted = [
                {"bbox_norm": s.bbox_norm, "kind": s.kind, "reading_order": s.reading_order}
                for s in spans if s.page == int(page_num)
            ]
            metrics = ingestion_metrics(gold_spans, predicted)
            print(f"  Page {page_num}: IoU={metrics['mean_iou']:.3f}  "
                  f"Layout={metrics['layout_accuracy']:.3f}  "
                  f"Coverage={metrics['coverage']:.3f}  "
                  f"Spurious={metrics['spurious_rate']:.3f}")


def main():
    app()
