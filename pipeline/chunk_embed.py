"""
pipeline/chunk_embed.py  —  Phase 2
Uses pipeline/embeddings.py and pipeline/vectorstore.py.
"""
import warnings, urllib3
warnings.filterwarnings("ignore", category=urllib3.exceptions.NotOpenSSLWarning)

import os, uuid
from typing import List, Dict
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

# Import from sibling modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.embeddings  import embed_texts
from pipeline.vectorstore import add_chunks


# ── Splitter ──────────────────────────────────────────────────────────────────

def _get_splitter():
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    return RecursiveCharacterTextSplitter(
        chunk_size    = 500,
        chunk_overlap = 50,
        separators    = ["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )


# ── Chunking strategies ───────────────────────────────────────────────────────

def chunk_prose(text: str, doc_meta: Dict) -> List[Dict]:
    raw = _get_splitter().split_text(text)
    return [
        {"chunk_id": str(uuid.uuid4()), "content": c.strip(),
         "section": f"passage_{i+1}", "page_or_row": i+1, **doc_meta}
        for i, c in enumerate(raw) if c.strip()
    ]


def chunk_spreadsheet(file_path: str, doc_meta: Dict) -> List[Dict]:
    df = pd.read_excel(file_path, engine="openpyxl")
    chunks = []
    for i, row in df.iterrows():
        parts = [f"{col}: {val}" for col, val in row.items() if pd.notna(val) and str(val).strip()]
        if not parts: continue
        chunks.append({
            "chunk_id":    str(uuid.uuid4()),
            "content":     f"[Row {i+1}] " + ", ".join(parts),
            "section":     "spreadsheet_row",
            "page_or_row": i+1,
            **doc_meta,
        })
    return chunks


def chunk_email(text: str, doc_meta: Dict) -> List[Dict]:
    lines = text.splitlines()
    header, body, in_body = [], [], False
    for line in lines:
        if not in_body and not line.strip(): in_body = True; continue
        (body if in_body else header).append(line)
    chunks = []
    if header:
        chunks.append({"chunk_id": str(uuid.uuid4()),
                       "content": "[Email headers] " + "\n".join(header),
                       "section": "email_header", "page_or_row": 0, **doc_meta})
    for c in chunk_prose("\n".join(body), doc_meta):
        c["section"] = "email_body"
        chunks.append(c)
    return chunks


# ── Embed wrapper (kept for backwards compat with retrieve.py) ────────────────

def embed(texts: List[str]) -> List[List[float]]:
    return embed_texts(texts)


# ── Main Phase 2 entry point ──────────────────────────────────────────────────

def run_phase2(
    file_path:   str,
    source_type: str,
    document_id: str,
    customer_id: str,
    doc_date:    str,
    filename:    str,
    db_session   = None,
) -> int:
    doc_meta = {
        "document_id": document_id,
        "customer_id": customer_id,
        "doc_date":    doc_date,
        "source_type": source_type,
        "filename":    filename,
    }

    # Parse + chunk
    if source_type == "xlsx":
        chunks = chunk_spreadsheet(file_path, doc_meta)
    elif source_type == "pdf":
        try:
            from liteparse import LiteParse
            text = LiteParse().parse(file_path).text
        except Exception:
            with open(file_path, "rb") as f:
                text = f.read().decode("utf-8", errors="replace")
        chunks = chunk_prose(text, doc_meta)
    else:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        chunks = chunk_email(text, doc_meta)

    if not chunks:
        print(f"[phase2] No chunks for {filename}")
        return 0

    # Embed (batched)
    all_vecs = []
    for i in range(0, len(chunks), 200):
        all_vecs.extend(embed_texts([c["content"] for c in chunks[i:i+200]]))

    # Write to vector store
    add_chunks(
        ids        = [c["chunk_id"] for c in chunks],
        texts      = [c["content"]  for c in chunks],
        embeddings = all_vecs,
        metadatas  = [{
            "document_id": c["document_id"],
            "customer_id": c["customer_id"],
            "doc_date":    c["doc_date"],
            "source_type": c["source_type"],
            "filename":    c["filename"],
            "section":     c["section"],
            "page_or_row": int(c["page_or_row"]),
        } for c in chunks],
    )

    print(f"[phase2] {len(chunks)} chunks indexed for '{filename}'")
    return len(chunks)