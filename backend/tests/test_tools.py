import json
from unittest.mock import patch, MagicMock
from metis.core.tools import ToolRegistry, make_rag_retrieve_tool, make_web_search_tool
from metis.core.schema import Evidence


def test_registry_get_tool_defs():
    registry = ToolRegistry()
    registry.register(
        name="test_tool",
        description="A test tool",
        parameters={"type": "object", "properties": {}},
        fn=lambda **kwargs: "result",
    )
    defs = registry.tool_defs()
    assert len(defs) == 1
    assert defs[0].name == "test_tool"


def test_registry_call():
    registry = ToolRegistry()
    registry.register(
        name="echo",
        description="Echo input",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}},
        fn=lambda text="": text,
    )
    result = registry.call("echo", {"text": "hello"})
    assert result == "hello"


def test_registry_call_unknown_tool():
    registry = ToolRegistry()
    result = registry.call("nonexistent", {})
    parsed = json.loads(result)
    assert "error" in parsed


def test_rag_retrieve_tool_formats_evidence():
    mock_evidence = [
        Evidence(span_id="s1", page=0, bbox_norm=(0.1, 0.2, 0.3, 0.4), text="some text", score=0.95),
    ]
    with patch("metis.core.tools.retrieve_semantic", return_value=mock_evidence):
        _, fn = make_rag_retrieve_tool("sha256:abc123")
        result = fn(query="test query", top_k=5)
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["text"] == "some text"
        assert parsed[0]["page"] == 0
        assert parsed[0]["score"] == 0.95
        assert parsed[0]["bbox_norm"] == [0.1, 0.2, 0.3, 0.4]

def test_web_search_tool_formats_results():
    mock_response = MagicMock()
    mock_response.results = [
        MagicMock(title="Result 1", url="https://example.com", content="Snippet 1"),
    ]
    with patch("metis.core.tools.TavilyClient") as MockTavily:
        MockTavily.return_value.search.return_value = mock_response
        _, fn = make_web_search_tool(api_key="test-key")
        result = fn(query="test query", max_results=3)
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["title"] == "Result 1"
        assert parsed[0]["url"] == "https://example.com"
        assert parsed[0]["snippet"] == "Snippet 1"
