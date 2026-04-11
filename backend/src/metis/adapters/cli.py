import json
import logging
from enum import Enum
from typing import Dict

import typer
import pymupdf
from rich import print
from pathlib import Path
from ..core.ingest import ingest_pdf_bytes, ingest_pdf_bytes_layout, ingest_pdf_bytes_tree
from ..core.retrieve import retrieve
from ..core.store import paths, read_spans_jsonl
from ..core.vectorize import vectorize_spans, retrieve_semantic, retrieve_hybrid
from ..core.agent import run_agent
from ..core.llm import AnthropicModel, OpenAIModel, OpenRouterModel, StreamEvent
from ..core.toc import render_toc
from ..core.tools import (
    ToolRegistry,
    make_locate_tool,
    make_rag_retrieve_tool,
    make_read_page_tool,
    make_read_section_tool,
    make_web_search_tool,
)
from ..core.tree_store import read_tree
from ..core.prompts import SYSTEM_PROMPT, build_system_prompt
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
    tree = "tree"
    blocks = "blocks"
    layout = "layout"


@app.command()
def ingest(
    pdf: Path,
    engine: Engine = typer.Option(Engine.tree, "--engine", help="Ingestion engine: 'tree' (default, new), 'layout', or 'blocks'"),
    parser: str = typer.Option(None, "--parser", help="Tree-mode parser override: 'docling' or 'pymupdf4llm'"),
    extract_words: bool = typer.Option(True, "--extract-words/--no-extract-words", help="Extract word-level bboxes (layout only)"),
    write_images: bool = typer.Option(True, "--write-images/--no-write-images", help="Materialize images (layout only)"),
    dpi: int = typer.Option(200, "--dpi", help="Image DPI (layout only)"),
):
    # Enable info logging so engine counts are visible
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if engine == Engine.tree:
        meta = ingest_pdf_bytes_tree(
            pdf.read_bytes(),
            source_filename=pdf.name,
            parser_name=parser,
        )
    elif engine == Engine.layout:
        meta = ingest_pdf_bytes_layout(
            pdf.read_bytes(),
            extract_words=extract_words,
            write_images=write_images,
            dpi=dpi,
            source_filename=pdf.name,
        )
    else:
        meta = ingest_pdf_bytes(pdf.read_bytes(), source_filename=pdf.name)
    print(meta)


@app.command("ls")
def list_docs(
    full: bool = typer.Option(False, "--full", "-f", help="Show full doc_id hash"),
):
    """List all ingested documents and their status."""
    import orjson
    from rich.table import Table
    from ..settings import DATA_DIR

    doc_files = sorted(DATA_DIR.glob("*.doc.json"))
    if not doc_files:
        print("[dim]No documents found in DATA_DIR.[/dim]")
        return

    entries = []
    for doc_file in doc_files:
        meta = orjson.loads(doc_file.read_bytes())
        doc_id = meta["doc_id"]
        p = paths(doc_id)

        # Display name: source_filename > PDF title > "—"
        name = meta.get("source_filename")
        if not name:
            try:
                import pymupdf
                pdf_doc = pymupdf.open(p["pdf"])
                pdf_title = pdf_doc.metadata.get("title", "").strip()
                pdf_doc.close()
                if pdf_title:
                    name = pdf_title
            except Exception:
                pass

        has_spans = p["spans"].exists()
        has_embeddings = p["embeddings"].exists()
        entries.append((doc_id, name, meta, has_spans, has_embeddings))

    if full:
        # List format: one doc per block, full hash visible
        for i, (doc_id, name, meta, has_spans, has_embeddings) in enumerate(entries):
            if i > 0:
                print()
            display_name = name or "—"
            ingested = "[green]✓ ingested[/green]" if has_spans else "[red]✗ not ingested[/red]"
            vectorized = "[green]✓ vectorized[/green]" if has_embeddings else "[red]✗ not vectorized[/red]"
            print(f"[bold]{display_name}[/bold]")
            print(f"  [cyan]{doc_id}[/cyan]")
            print(f"  {meta.get('n_pages', '?')} pages, {meta.get('n_spans', '?')} spans  |  {ingested}  |  {vectorized}")
    else:
        # Table format with truncated hash
        table = Table(title="Documents")
        table.add_column("doc_id", style="cyan", no_wrap=True)
        table.add_column("name", style="bold")
        table.add_column("pages", justify="right")
        table.add_column("spans", justify="right")
        table.add_column("ingested", justify="center")
        table.add_column("vectorized", justify="center")

        for doc_id, name, meta, has_spans, has_embeddings in entries:
            short_id = doc_id[:19]  # "sha256:" + 12 hex chars
            ingested = "[green]✓[/green]" if has_spans else "[red]✗[/red]"
            vectorized = "[green]✓[/green]" if has_embeddings else "[red]✗[/red]"
            table.add_row(
                short_id,
                name or "[dim]—[/dim]",
                str(meta.get("n_pages", "?")),
                str(meta.get("n_spans", "?")),
                ingested,
                vectorized,
            )
        print(table)


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

@app.command("retrieve")
def retrieve_hybrid_cmd(
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
    ev = retrieve_hybrid(doc_id=doc_id, query=query, **kwargs)
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
        elif prov == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY", "")

    if not api_key:
        console.print("[red]No API key found. Set METIS_LLM_API_KEY or ANTHROPIC_API_KEY / OPENAI_API_KEY / OPENROUTER_API_KEY.[/red]")
        raise typer.Exit(1)

    # Validate doc exists and has embeddings. Accept either the new tree
    # pipeline (tree.json + embeddings_v2.npy) or the legacy span pipeline
    # (spans.jsonl + embeddings.npy).
    p = paths(doc_id)
    has_tree = p["tree"].exists()
    has_spans = p["spans"].exists()
    if not (has_tree or has_spans):
        console.print(f"[red]Document not found: {doc_id}[/red]")
        console.print("Run: metis ingest <pdf> first.")
        raise typer.Exit(1)

    if has_tree:
        if not p["embeddings_v2"].exists():
            console.print(f"[yellow]No embeddings found for this document.[/yellow]")
            console.print("Run the tree embedding pipeline first.")
            raise typer.Exit(1)
    else:
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
    elif prov == "openrouter":
        llm = OpenRouterModel(api_key=api_key, model=mod, temperature=AGENT_TEMPERATURE)
    else:
        console.print(f"[red]Unknown provider: {prov}. Use 'anthropic', 'openai', or 'openrouter'.[/red]")
        raise typer.Exit(1)

    # Build tools. If a tree sidecar exists we use the hierarchical tools
    # (locate + read_section) and a TOC-aware system prompt. Otherwise we
    # fall back to the legacy span-based RAG tool + flat system prompt.
    registry = ToolRegistry()
    tree_path = p["tree"]
    if tree_path.exists():
        loc_def, loc_fn = make_locate_tool(doc_id)
        registry.register(loc_def.name, loc_def.description, loc_def.parameters, loc_fn)

        rs_def, rs_fn = make_read_section_tool(doc_id)
        registry.register(rs_def.name, rs_def.description, rs_def.parameters, rs_fn)

        system_prompt = build_system_prompt(render_toc(read_tree(tree_path)))
    else:
        rag_def, rag_fn = make_rag_retrieve_tool(doc_id)
        registry.register(rag_def.name, rag_def.description, rag_def.parameters, rag_fn)

        system_prompt = SYSTEM_PROMPT

    if p["page_md"].exists():
        rp_def, rp_fn = make_read_page_tool(doc_id)
        registry.register(rp_def.name, rp_def.description, rp_def.parameters, rp_fn)

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

        if tool_name == "read_page":
            # Show first 200 chars of page text
            display = result_str[:200] + "..." if len (result_str) > 200 else result_str
            console.print(f"    [dim yellow]{display}[/dim yellow]")
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
                doc_id=doc_id,
                user_query=user_input,
                tools=registry,
                system_prompt=system_prompt,
                max_iterations=AGENT_MAX_ITER,
                on_stream=on_stream,
                on_tool_result=on_tool_result,
            )
            console.print()  # newline after streamed response
            console.print()  # blank line before next prompt

    except KeyboardInterrupt:
        console.print("\n[dim]Goodbye.[/dim]")


@app.command("eval")
def eval_cmd(
    eval_set: str = typer.Option(
        "backend/tests/eval/retrieval_eval.jsonl",
        "--eval-set",
        help="Path to the JSONL eval set",
    ),
    papers_dir: str = typer.Option(
        "backend/tests/eval/papers",
        "--papers-dir",
        help="Directory containing eval PDFs referenced in the eval set",
    ),
    output_dir: str = typer.Option("eval_results", "--output", help="Output directory"),
    system: str = typer.Option("new", "--system", help="System name tag (for the output file)"),
):
    """Run the retrieval eval harness against the current code."""
    from pathlib import Path
    import orjson
    from metis.core.eval import run_eval
    from metis.core.ingest import ingest_pdf_bytes_tree
    from metis.core.embed_v2 import vectorize_tree
    from metis.core.retrieve_v2 import retrieve_paragraphs

    papers_root = Path(papers_dir)
    paper_to_doc_id: Dict[str, str] = {}
    with Path(eval_set).open("rb") as f:
        for line in f:
            if not line.strip():
                continue
            row = orjson.loads(line)
            name = row["paper"]
            if name in paper_to_doc_id:
                continue
            pdf = papers_root / name
            if not pdf.exists():
                typer.echo(f"Missing paper: {pdf}", err=True)
                continue
            meta = ingest_pdf_bytes_tree(pdf.read_bytes())
            vectorize_tree(meta["doc_id"])
            paper_to_doc_id[name] = meta["doc_id"]

    def _runner(doc_id_or_paper: str, query: str):
        doc_id = paper_to_doc_id.get(doc_id_or_paper, doc_id_or_paper)
        hits = retrieve_paragraphs(doc_id, query, top_k=10)
        ranked = [h["para_id"] for h in hits]
        answer = hits[0]["preview"] if hits else ""
        return answer, ranked

    summary = run_eval(
        eval_set_path=eval_set,
        output_dir=output_dir,
        system_name=system,
        agent_runner=_runner,
    )
    typer.echo(f"Eval complete: {summary}")


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
