"""
Loads the local embedding model once, and provides a function to convert
text into embeddings. Using sentence-transformers means this runs entirely
on your laptop -- no API calls, no quota limits.
"""

from sentence_transformers import SentenceTransformer

_model = None  # loaded once, reused for every call (loading is slow)

def get_embedding_model(model_name: str = "BAAI/bge-small-en-v1.5"):
    global _model
    if _model is None:
        print(f"Loading embedding model '{model_name}' (first time only, "
              f"may take a minute to download)...")
        _model = SentenceTransformer(model_name)
    return _model

def embed_texts(texts: list[str], model_name: str = "BAAI/bge-small-en-v1.5") -> list[list[float]]:
    """Converts a list of text strings into a list of embedding vectors."""
    model = get_embedding_model(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings.tolist()