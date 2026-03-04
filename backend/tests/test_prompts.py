from metis.core.prompts import SYSTEM_PROMPT, format_query_with_selections


def test_no_selections_passthrough():
    assert format_query_with_selections("hello", []) == "hello"


def test_no_selections_none():
    assert format_query_with_selections("hello", None) == "hello"


def test_single_selection_formatted():
    spans = [{"page": 2, "text": "Some text here", "bbox_norm": (0.1, 0.2, 0.3, 0.4), "span_id": "s1", "iou": 0.85}]
    result = format_query_with_selections("What does this mean?", spans)
    assert "selected the following region" in result
    assert "Page 2" in result
    assert "0.85" in result
    assert "Some text here" in result
    assert "What does this mean?" in result


def test_multiple_selections_formatted():
    spans = [
        {"page": 0, "text": "First", "bbox_norm": (0, 0, 1, 1), "span_id": "s1", "iou": 0.9},
        {"page": 1, "text": "Second", "bbox_norm": (0, 0, 1, 1), "span_id": "s2", "iou": 0.7},
    ]
    result = format_query_with_selections("Explain", spans)
    assert "First" in result
    assert "Second" in result
    assert result.index("First") < result.index("Second")


def test_system_prompt_mentions_selections():
    assert "selected region" in SYSTEM_PROMPT.lower()
