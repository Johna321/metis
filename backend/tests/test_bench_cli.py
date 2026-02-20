from typer.testing import CliRunner
from metis.adapters.cli import app

runner = CliRunner()


def test_benchmark_help():
    result = runner.invoke(app, ["benchmark", "--help"])
    assert result.exit_code == 0
    assert "retrieval" in result.output.lower()
