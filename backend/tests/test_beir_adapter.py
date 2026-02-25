import numpy as np
from metis.benchmark.beir_adapter import MetisDenseRetriever


def test_encode_queries_returns_ndarray():
    model = MetisDenseRetriever(model_name="all-MiniLM-L6-v2")
    queries = ["What is attention?", "How do transformers work?"]
    embeddings = model.encode_queries(queries, batch_size=2)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] == 384  # all-MiniLM-L6-v2 dim


def test_encode_corpus_returns_ndarray():
    model = MetisDenseRetriever(model_name="all-MiniLM-L6-v2")
    corpus = [
        {"title": "Attention", "text": "Attention is a mechanism in neural networks."},
        {"title": "", "text": "Transformers use self-attention for sequence modeling."},
    ]
    embeddings = model.encode_corpus(corpus, batch_size=2)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] == 384


def test_embeddings_are_normalized():
    model = MetisDenseRetriever(model_name="all-MiniLM-L6-v2")
    queries = ["test query"]
    embeddings = model.encode_queries(queries, batch_size=1)
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)
