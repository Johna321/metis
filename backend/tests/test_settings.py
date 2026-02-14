from metis.settings import EMBED_MODEL

def test_embed_model_default():
    assert EMBED_MODEL == "all-MiniLM-L6-v2"
