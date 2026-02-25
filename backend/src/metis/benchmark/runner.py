from __future__ import annotations
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from ..settings import DATA_DIR

log = logging.getLogger(__name__)

BEIR_DATA_DIR = DATA_DIR / "beir_datasets"
RESULTS_DIR = DATA_DIR / "benchmark_results"

AVAILABLE_DATASETS = ("scifact", "nfcorpus", "scidocs")


@dataclass
class BenchmarkResult:
    dataset: str
    model: str
    ndcg: dict
    map_score: dict
    recall: dict
    precision: dict
    encoding_time_s: float = 0.0
    retrieval_time_s: float = 0.0


def save_result(result: BenchmarkResult, output_dir: Path | None = None) -> Path:
    output_dir = output_dir or RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{ts}_{result.dataset}_{result.model}.json"
    path = output_dir / filename
    data = asdict(result)
    data["timestamp"] = ts
    path.write_text(json.dumps(data, indent=2))
    return path


def run_retrieval_benchmark(
    dataset_name: str = "scifact",
    model_name: str | None = None,
    split: str = "test",
) -> BenchmarkResult:
    """Download a BEIR dataset, run dense retrieval, evaluate, save results."""
    import time
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader
    from beir.retrieval.evaluation import EvaluateRetrieval
    from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
    from .beir_adapter import MetisDenseRetriever
    from ..settings import EMBED_MODEL

    model_name = model_name or EMBED_MODEL

    # 1. Download dataset
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    data_path = util.download_and_unzip(url, str(BEIR_DATA_DIR))
    corpus, queries, qrels = GenericDataLoader(data_path).load(split=split)
    log.info("Loaded %s: %d docs, %d queries", dataset_name, len(corpus), len(queries))

    # 2. Set up retriever
    retriever_model = MetisDenseRetriever(model_name=model_name)
    dres = DRES(retriever_model, batch_size=64)
    evaluator = EvaluateRetrieval(dres, score_function="cos_sim")

    # 3. Retrieve
    t0 = time.time()
    results = evaluator.retrieve(corpus, queries)
    retrieval_time = time.time() - t0

    # 4. Evaluate
    ndcg, map_score, recall, precision = evaluator.evaluate(qrels, results, evaluator.k_values)
    log.info("nDCG@10: %.4f | MAP@10: %.4f | Recall@100: %.4f",
             ndcg.get("NDCG@10", 0), map_score.get("MAP@10", 0), recall.get("Recall@100", 0))

    result = BenchmarkResult(
        dataset=dataset_name,
        model=model_name,
        ndcg=ndcg,
        map_score=map_score,
        recall=recall,
        precision=precision,
        retrieval_time_s=round(retrieval_time, 2),
    )
    save_result(result)
    return result
