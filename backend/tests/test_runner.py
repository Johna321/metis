import json
from pathlib import Path
from metis.benchmark.runner import BenchmarkResult, save_result


def test_save_result_writes_json(tmp_path):
    result = BenchmarkResult(
        dataset="scifact",
        model="all-MiniLM-L6-v2",
        ndcg={"NDCG@10": 0.55},
        map_score={"MAP@10": 0.40},
        recall={"Recall@100": 0.80},
        precision={"P@10": 0.35},
    )
    path = save_result(result, output_dir=tmp_path)
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["dataset"] == "scifact"
    assert data["model"] == "all-MiniLM-L6-v2"
    assert "ndcg" in data
    assert "timestamp" in data


def test_save_result_filename_contains_dataset(tmp_path):
    result = BenchmarkResult(
        dataset="nfcorpus",
        model="all-MiniLM-L6-v2",
        ndcg={}, map_score={}, recall={}, precision={},
    )
    path = save_result(result, output_dir=tmp_path)
    assert "nfcorpus" in path.name
