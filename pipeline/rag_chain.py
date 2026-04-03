"""
pipeline/rag_chain.py
Orchestration backbone: LangChain ConversationalRetrievalChain.
Flow: query → embed → vectorstore → rerank → conflict → LLM → cite → CRM
"""
import os, json, uuid, time
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
load_dotenv()

LLM_MODEL  = os.getenv("LLM_MODEL",  "gpt-4o")
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_KEY   = os.getenv("GROQ_API_KEY",   "")


# ── Citation prompt ───────────────────────────────────────────────────────────

CITATION_PROMPT = """You are a precise knowledge assistant for a small business.
Answer the employee's query using ONLY the provided source documents.
{conflict_note}
After EVERY factual claim, add a citation: [SOURCE: filename, section]
If sources don't contain enough info, say so honestly.

SOURCES:
{sources}

QUERY: {query}

You MUST reply with ONLY valid JSON — no prose before or after:
{{
  "answer": "Full answer with inline [SOURCE: filename, section] citations.",
  "citations": [
    {{
      "filename": "exact filename",
      "section": "section or chunk ref",
      "doc_date": "YYYY-MM-DD",
      "excerpt": "short verbatim excerpt supporting the claim"
    }}
  ],
  "conflict_detected": true or false,
  "conflict_explanation": "One sentence explanation, or empty string."
}}"""


def _format_sources(chunks: List[Dict]) -> str:
    out = ""
    for i, c in enumerate(chunks[:6]):
        m = c["metadata"]
        out += (
            f"\n--- SOURCE {i+1} ---\n"
            f"File: {m.get('filename','unknown')}\n"
            f"Section: {m.get('section','—')}\n"
            f"Date: {m.get('doc_date','unknown')}\n"
            f"Content:\n{c['content']}\n"
        )
    return out


def _call_llm(prompt: str) -> Dict:
    """Call LLM (OpenAI or Groq) and parse JSON response."""
    t0 = time.time()

    if LLM_MODEL.lower() == "groq":
        # Use Groq API
        import groq
        client = groq.Groq(api_key=GROQ_KEY)
        rsp = client.chat.completions.create(
            model       = "llama-3.1-8b-instant",  # Fast free model
            messages    = [{"role":"user","content":prompt}],
            temperature = 0.1,
            max_tokens  = 1200,
        )
    else:
        # Use OpenAI API
        from openai import OpenAI
        rsp = OpenAI(api_key=OPENAI_KEY).chat.completions.create(
            model       = LLM_MODEL,
            messages    = [{"role":"user","content":prompt}],
            temperature = 0.1,
            max_tokens  = 1200,
        )

    raw = rsp.choices[0].message.content.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        parts = raw.split("```")
        raw   = parts[1] if len(parts)>1 else raw
        raw   = raw[4:] if raw.startswith("json") else raw
    raw = raw.strip()
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {
            "answer":               raw,
            "citations":            [],
            "conflict_detected":    False,
            "conflict_explanation": "",
        }
    result["latency_ms"] = round((time.time()-t0)*1000, 1)
    return result


def _mock_answer(query: str, chunks: List[Dict], conflict: Optional[Dict]) -> Dict:
    """Fallback when no OpenAI key is set."""
    c = chunks[0] if chunks else {}
    return {
        "answer": (
            "[MOCK — set OPENAI_API_KEY or GROQ_API_KEY in .env for real answers]\n"
            f"Best match from '{c.get('metadata',{}).get('filename','?')}': "
            f"{c.get('content','')[:300]}…"
        ),
        "citations": [
            {
                "filename": x.get("metadata",{}).get("filename","?"),
                "section":  x.get("metadata",{}).get("section","—"),
                "doc_date": x.get("metadata",{}).get("doc_date","—"),
                "excerpt":  x.get("content","")[:120],
            }
            for x in chunks[:3]
        ],
        "conflict_detected":    conflict is not None,
        "conflict_explanation": conflict["explanation"] if conflict else "",
        "latency_ms":           0,
    }


# ── Main RAG entry point ──────────────────────────────────────────────────────

def run_rag(
    query:        str,
    customer_id:  str,
    customer_name: str,
    db_session    = None,
    chat_history: List[Tuple[str,str]] = None,
) -> Dict:
    """
    Full RAG pipeline in one call.

    Steps:
      1. Embed query
      2. Retrieve top-k chunks from vector store (customer-scoped)
      3. Re-rank (if RERANK_BACKEND != none)
      4. Detect conflicts
      5. Generate answer (or mock)
      6. Save Ticket to DB + CRM (simplified for now)
      7. Return structured result

    Returns dict with keys:
      answer, citations, conflict_detected, conflict_explanation,
      ticket_id, latency_ms
    """
    from pipeline.embeddings  import embed_query
    from pipeline.vectorstore import query_chunks
    from pipeline.rerank      import rerank
    from pipeline.conflict    import detect
    from pipeline.crm         import build_ticket, save_ticket

    # 1. Embed
    qvec = embed_query(query)

    # 2. Retrieve
    chunks = query_chunks(qvec, customer_id, top_k=10)
    if not chunks:
        return {
            "answer":               "No relevant documents found. Please upload files first.",
            "citations":            [],
            "conflict_detected":    False,
            "conflict_explanation": "",
            "ticket_id":            None,
            "latency_ms":           0,
        }

    # 3. Re-rank
    chunks = rerank(query, chunks, top_n=6)

    # 4. Conflict detection
    conflict = detect(chunks)

    # 5. Generate answer
    conflict_note = (
        f"\nIMPORTANT — CONFLICT DETECTED:\n{conflict['explanation']}\n"
        f"You MUST mention the conflict and state which source you trust.\n"
    ) if conflict else ""

    prompt = CITATION_PROMPT.format(
        conflict_note = conflict_note,
        sources       = _format_sources(chunks),
        query         = query,
    )

    t0     = time.time()
    has_llm_key = OPENAI_KEY or (LLM_MODEL.lower() == "groq" and GROQ_KEY)
    result = _call_llm(prompt) if has_llm_key else _mock_answer(query, chunks, conflict)
    if "latency_ms" not in result:
        result["latency_ms"] = round((time.time()-t0)*1000, 1)

    # 6. Save to mock CRM (simplified - no DB for now)
    ticket_id = str(uuid.uuid4())
    result["ticket_id"] = ticket_id

    ticket = build_ticket(ticket_id, customer_name, query, result)
    save_ticket(ticket)

    return result