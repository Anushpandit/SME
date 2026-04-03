"""
pipeline/embeddings.py
Primary  : OpenAI text-embedding-3-small (1536 dims)
Backup   : sentence-transformers/all-MiniLM-L6-v2 (local, free)
Switch via EMBEDDING_BACKEND=openai|local in .env
"""
import os
from typing import List
from dotenv import load_dotenv
load_dotenv()

BACKEND = os.getenv("EMBEDDING_BACKEND", "openai")
_st_model = None

def _load_st():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[embeddings] Loaded local sentence-transformers model.")
    return _st_model

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of strings. Returns list of float vectors."""
    if not texts:
        return []
    if BACKEND == "local":
        model = _load_st()
        return model.encode(texts, show_progress_bar=False).tolist()
    # OpenAI
    from openai import OpenAI
    client   = OpenAI()
    response = client.embeddings.create(
        model = "text-embedding-3-small",
        input = texts,
    )
    return [item.embedding for item in response.data]

def embed_query(text: str) -> List[float]:
    """Embed a single query string."""
    return embed_texts([text])[0]