from typer.testing import CliRunner
from metis.adapters.cli import app

runner = CliRunner()

def test_vectorize_command_exists():
    result = runner.invoke(app, ["vectorize", "--help"])
    assert result.exit_code == 0
    assert "doc_id" in result.output.lower() or "DOC_ID" in result.output

def test_retrieve_semantic_command_exists():
    result = runner.invoke(app, ["retrieve-semantic", "--help"])
    assert result.exit_code == 0
