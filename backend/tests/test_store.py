from metis.core.store import paths

def test_paths_has_embeddings_keys():
    p = paths("sha256:abc123")
    assert "embeddings" in p
    assert "embeddings_meta" in p
    assert str(p["embeddings"]).endswith(".embeddings.npy")
    assert str(p["embeddings_meta"]).endswith(".embeddings_meta.json")
