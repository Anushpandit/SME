"""
pipeline/rerank.py
Primary  : Cohere rerank-english-v3.0
Backup   : cross-encoder/ms-marco-MiniLM-L-6-v2 (local, free)
Switch via RERANK_BACKEND=cohere|local|none in .env
"""
import os
from typing import List, Dict
from dotenv import load_dotenv
load_dotenv()

BACKEND    = os.getenv("RERANK_BACKEND",  "none")   # set to "cohere" or "local" to enable
COHERE_KEY = os.getenv("COHERE_API_KEY",  "")
_ce_model  = None


def _load_cross_encoder():
    global _ce_model
    if _ce_model is None:
        from sentence_transformers import CrossEncoder
        _ce_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("[rerank] Loaded local cross-encoder.")
    return _ce_model


def rerank(query: str, chunks: List[Dict], top_n: int = 6) -> List[Dict]:
    """
    Re-rank retrieved chunks by relevance to query.
    Returns top_n chunks sorted best-first.
    Falls through (returns input unchanged) if RERANK_BACKEND=none.
    """
    if not chunks or BACKEND == "none":
        return chunks[:top_n]

    if BACKEND == "cohere" and COHERE_KEY:
        import cohere
        co      = cohere.Client(COHERE_KEY)
        docs    = [c["content"] for c in chunks]
        results = co.rerank(
            model     = "rerank-english-v3.0",
            query     = query,
            documents = docs,
            top_n     = top_n,
        )
        reranked = [chunks[r.index] for r in results.results]
        print(f"[rerank] Cohere reranked {len(chunks)} → {len(reranked)} chunks")
        return reranked

    if BACKEND == "local":
        ce     = _load_cross_encoder()
        pairs  = [(query, c["content"]) for c in chunks]
        scores = ce.predict(pairs).tolist()
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        reranked = [c for _, c in ranked[:top_n]]
        print(f"[rerank] Cross-encoder reranked {len(chunks)} → {len(reranked)} chunks")
        return reranked

    return chunks[:top_n]