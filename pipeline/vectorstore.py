"""
pipeline/vectorstore.py
Primary  : Chroma  (local persistent, zero config)
Backup   : FAISS   (in-memory, no server needed)
Switch via VECTOR_BACKEND=chroma|faiss in .env
"""
import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
load_dotenv()

BACKEND     = os.getenv("VECTOR_BACKEND", "chroma")
CHROMA_PATH = os.getenv("CHROMA_PATH",    "./chroma_db")
COLLECTION  = "sme_knowledge"

_chroma_col  = None
_faiss_index = None
_faiss_meta  = []   # parallel list of metadata dicts for FAISS


# ── Chroma ────────────────────────────────────────────────────────────────────

def _get_chroma():
    global _chroma_col
    if _chroma_col is None:
        import chromadb
        client      = chromadb.PersistentClient(path=CHROMA_PATH)
        _chroma_col = client.get_or_create_collection(
            name     = COLLECTION,
            metadata = {"hnsw:space": "cosine"},
        )
    return _chroma_col


# ── FAISS ─────────────────────────────────────────────────────────────────────

def _get_faiss(dim: int = 384):
    global _faiss_index
    if _faiss_index is None:
        import faiss
        _faiss_index = faiss.IndexFlatIP(dim)   # inner-product = cosine on normalised vecs
        print(f"[vectorstore] FAISS index created (dim={dim})")
    return _faiss_index


def _normalise(vec: List[float]) -> List[float]:
    import math
    mag = math.sqrt(sum(x*x for x in vec)) or 1.0
    return [x/mag for x in vec]


# ── Public API ────────────────────────────────────────────────────────────────

def add_chunks(
    ids:        List[str],
    texts:      List[str],
    embeddings: List[List[float]],
    metadatas:  List[Dict],
) -> None:
    """Add chunks to the vector store."""
    if BACKEND == "faiss":
        import numpy as np
        global _faiss_meta
        normed = [_normalise(e) for e in embeddings]
        _get_faiss(len(normed[0])).add(np.array(normed, dtype="float32"))
        _faiss_meta.extend(metadatas)
        return
    # Chroma — metadata values must be str/int/float
    def _safe(m):
        return {k: str(v) if not isinstance(v, (int, float)) else v for k,v in m.items()}
    BATCH = 500
    for i in range(0, len(ids), BATCH):
        _get_chroma().add(
            ids        = ids[i:i+BATCH],
            documents  = texts[i:i+BATCH],
            embeddings = embeddings[i:i+BATCH],
            metadatas  = [_safe(m) for m in metadatas[i:i+BATCH]],
        )


def query_chunks(
    query_embedding: List[float],
    customer_id:     str,
    top_k:           int = 8,
) -> List[Dict]:
    """
    Semantic search. Returns list of dicts:
    { chunk_id, content, metadata, similarity }
    """
    if BACKEND == "faiss":
        import numpy as np
        global _faiss_meta
        idx   = _get_faiss()
        n     = min(top_k * 3, idx.ntotal)
        if n == 0: return []
        normed = np.array([_normalise(query_embedding)], dtype="float32")
        scores, positions = idx.search(normed, n)
        results = []
        for score, pos in zip(scores[0], positions[0]):
            if pos < 0 or pos >= len(_faiss_meta): continue
            meta = _faiss_meta[pos]
            if meta.get("customer_id") != customer_id: continue
            results.append({
                "chunk_id":   meta.get("chunk_id", str(pos)),
                "content":    meta.get("content",  ""),
                "metadata":   meta,
                "similarity": round(float(score), 4),
            })
            if len(results) >= top_k: break
        return results

    # Chroma
    col   = _get_chroma()
    total = col.count()
    if total == 0: return []
    res   = col.query(
        query_embeddings = [query_embedding],
        n_results        = min(top_k, total),
        where            = {"customer_id": customer_id},
        include          = ["documents","metadatas","embeddings","distances"],
    )
    chunks  = []
    embeds  = res.get("embeddings")
    if embeds and len(embeds) > 0:
        embeds = embeds[0]  # Get embeddings for the first (only) query
    else:
        embeds = []

    for i, (cid, doc, meta, dist) in enumerate(zip(
        res.get("ids",        [[]])[0],
        res.get("documents",  [[]])[0],
        res.get("metadatas",  [[]])[0],
        res.get("distances",  [[]])[0],
    )):
        embedding = embeds[i] if i < len(embeds) else []
        chunks.append({
            "chunk_id":   cid,
            "content":    doc,
            "metadata":   meta,
            "similarity": round(1 - dist, 4) if dist is not None else 0.0,
            "_embedding": embedding,
        })
    return chunks


def count() -> int:
    if BACKEND == "faiss":
        return _get_faiss().ntotal if _faiss_index else 0
    return _get_chroma().count()