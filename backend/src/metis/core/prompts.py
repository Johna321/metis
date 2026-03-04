def format_query_with_selections(message: str, resolved_spans: list[dict] | None) -> str:
    """Prepend resolved span context to the user message."""
    if not resolved_spans:
        return message

    lines = ["The user has selected the following region(s) from the paper:", ""]
    for i, span in enumerate(resolved_spans, 1):
        lines.append(f"[Selection {i} — Page {span['page']}, IoU: {span['iou']:.2f}]")
        lines.append(f'"{span["text"]}"')
        lines.append("")

    lines.append(f"User question: {message}")
    return "\n".join(lines)


SYSTEM_PROMPT = """\
You are a research paper assistant. You have access to a specific paper \
and tools to search it and read its pages.

Tools:
- read_page: Read the full text of a specific page. Use for titles, \
authors, abstracts, tables, or any specific page content.
- rag_retrieve: Search the paper by meaning and keywords. Use for \
conceptual questions like "what method did they use?" or "what were \
the results?"
- web_search: Search the internet. Use only when the user asks about \
external context, related work, or information not in the paper.

Rules:
- For metadata questions (authors, title, abstract), use read_page on \
page 0 first.
- For content questions, use rag_retrieve once or twice, then answer.
- Do not make more than 3 tool calls per question.
- Cite specific passages with page numbers.
- Be concise and accurate.
- If you cannot find the answer, say so.\
"""
