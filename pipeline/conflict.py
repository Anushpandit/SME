"""
pipeline/conflict.py
Step-by-step conflict detection:
  1. Group chunks by cosine similarity > 0.85 (same topic)
  2. Within each group, compare doc_date across different source files
  3. Flag contradictory pairs
  4. Ask LLM to explain the conflict and rank by date
  5. Return structured conflict report
"""
import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
load_dotenv()

SIMILARITY_THRESHOLD = float(os.getenv("CONFLICT_THRESHOLD", "0.85"))
OPENAI_KEY           = os.getenv("OPENAI_API_KEY", "")
GROQ_KEY             = os.getenv("GROQ_API_KEY", "")
LLM_MODEL            = os.getenv("LLM_MODEL", "gpt-4o")


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) == 0 or len(b) == 0: return 0.0
    dot = sum(x*y for x,y in zip(a,b))
    ma  = sum(x*x for x in a)**0.5
    mb  = sum(x*x for x in b)**0.5
    return dot/(ma*mb) if ma and mb else 0.0


def _ask_llm_about_conflict(chunk_a: Dict, chunk_b: Dict) -> str:
    """Ask LLM to describe what the two chunks disagree on."""
    has_llm_key = OPENAI_KEY or (LLM_MODEL.lower() == "groq" and GROQ_KEY)
    if not has_llm_key:
        return (
            f"Possible conflict: '{chunk_a['metadata']['filename']}' "
            f"(dated {chunk_a['metadata'].get('doc_date','?')}) and "
            f"'{chunk_b['metadata']['filename']}' "
            f"(dated {chunk_b['metadata'].get('doc_date','?')}) "
            f"cover the same topic with different content."
        )

    prompt = f"""Two document passages cover the same topic but may contradict each other.

Passage A — File: {chunk_a['metadata']['filename']}, Date: {chunk_a['metadata'].get('doc_date','unknown')}
\"\"\"{chunk_a['content']}\"\"\"

Passage B — File: {chunk_b['metadata']['filename']}, Date: {chunk_b['metadata'].get('doc_date','unknown')}
\"\"\"{chunk_b['content']}\"\"\"

In one sentence: what specific fact do they disagree on?
Then in one sentence: which is more recent and should be trusted?
Reply in plain text only, no JSON."""

    try:
        if LLM_MODEL.lower() == "groq":
            import groq
            client = groq.Groq(api_key=GROQ_KEY)
            r = client.chat.completions.create(
                model       = "llama-3.1-8b-instant",
                messages    = [{"role":"user","content":prompt}],
                temperature = 0,
                max_tokens  = 120,
            )
        else:
            from openai import OpenAI
            r = OpenAI(api_key=OPENAI_KEY).chat.completions.create(
                model       = LLM_MODEL,
                messages    = [{"role":"user","content":prompt}],
                temperature = 0,
                max_tokens  = 120,
            )
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f"Conflict detected (LLM explanation unavailable: {e})"


def detect(chunks: List[Dict]) -> Optional[Dict]:
    """
    Main entry point. Returns conflict report dict or None.

    Report keys:
      conflict, trusted, outdated, trusted_date, outdated_date,
      explanation, similarity
    """
    # Build embedding list (stored under _embedding key from vectorstore)
    vecs = [c.get("_embedding", []) for c in chunks]

    for i in range(len(chunks)):
        for j in range(i+1, len(chunks)):
            ci, cj = chunks[i], chunks[j]

            # Only flag conflicts between DIFFERENT source documents
            if ci["metadata"].get("document_id") == cj["metadata"].get("document_id"):
                continue

            # Skip trivially short chunks
            if len(ci["content"]) < 50 or len(cj["content"]) < 50:
                continue

            sim = _cosine(vecs[i], vecs[j])
            if sim < SIMILARITY_THRESHOLD:
                continue

            # Rank by doc_date — trust newer
            di = ci["metadata"].get("doc_date", "0000-00-00")
            dj = cj["metadata"].get("doc_date", "0000-00-00")
            trusted, outdated   = (ci, cj) if di >= dj else (cj, ci)
            trusted_date        = max(di, dj)
            outdated_date       = min(di, dj)

            # Ask LLM to explain the disagreement
            llm_note = _ask_llm_about_conflict(outdated, trusted)

            return {
                "conflict":       True,
                "similarity":     round(sim, 3),
                "trusted":        trusted,
                "outdated":       outdated,
                "trusted_date":   trusted_date,
                "outdated_date":  outdated_date,
                "explanation": (
                    f"Conflict detected between "
                    f"'{trusted['metadata']['filename']}' (dated {trusted_date}) "
                    f"and '{outdated['metadata']['filename']}' (dated {outdated_date}). "
                    f"Trusting the more recent document. {llm_note}"
                ),
            }
    return None