import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from pipeline.rag_chain import run_rag

st.set_page_config(page_title="SME Knowledge Agent", page_icon="🔍", layout="wide")

if "messages"    not in st.session_state: st.session_state.messages    = []
if "last_result" not in st.session_state: st.session_state.last_result = None
if "last_ticket" not in st.session_state: st.session_state.last_ticket = None
if "history"     not in st.session_state: st.session_state.history     = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### KnowledgeAgent")
    st.caption("SME Multi-Format RAG System")
    st.divider()

    # Mock customer selection for demo
    customers = ["Acme Corp", "TechStart Inc", "Global Solutions"]
    customer_name = st.selectbox("Customer scope", customers)
    customer_id = customer_name.lower().replace(" ", "_")

    st.divider()
    st.markdown("**Upload document**")
    uploaded = st.file_uploader("PDF, Excel, or text",
                                type=["pdf","xlsx","xls","txt","eml"],
                                label_visibility="collapsed")

    if uploaded:
        # Mock document processing
        st.success(f"Document '{uploaded.name}' would be processed here")
        # In real implementation, call pipeline.ingest.ingest_document()

    st.divider()
    import os
    backend = os.getenv("EMBEDDING_BACKEND","openai")
    rerank  = os.getenv("RERANK_BACKEND","none")
    st.caption(f"Embed: `{backend}` · Rerank: `{rerank}`")
    st.caption(f"LLM: `{'gpt-4o' if os.getenv('OPENAI_API_KEY') else 'mock'}`")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.last_result = None
        st.session_state.history = []
        st.rerun()

# ── Main columns ──────────────────────────────────────────────────────────────
col_chat, col_panel = st.columns([3, 1.2], gap="medium")

with col_chat:
    st.markdown("#### Knowledge chat")
    st.caption(f"Scope: **{customer_name}**")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if msg.get("conflict"):
                    st.warning(f"⚠️ {msg['conflict']}")
                for c in msg.get("citations",[])[:3]:
                    st.caption(f"📄 {c.get('filename','?')} · "
                               f"{c.get('section','—')} · {c.get('doc_date','—')}")

    query = st.chat_input("Ask a question about your documents...",
                          disabled=False)

    if query:
        st.session_state.messages.append({"role":"user","content":query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Searching, re-ranking, detecting conflicts..."):
                result = run_rag(
                    query         = query,
                    customer_id   = customer_id,
                    customer_name = customer_name,
                    db_session    = None,  # Simplified for demo
                    chat_history  = st.session_state.history,
                )

            st.markdown(result["answer"])
            if result.get("conflict_detected"):
                st.warning(f"⚠️ **Conflict** — {result.get('conflict_explanation','')}")
            for c in result.get("citations",[])[:3]:
                st.caption(f"📄 {c.get('filename','?')} · "
                           f"{c.get('section','—')} · {c.get('doc_date','—')}")
            st.caption(f"⏱ {result.get('latency_ms',0):.0f} ms")

        # Save to session
        st.session_state.history.append((query, result["answer"]))
        st.session_state.last_result = result
        st.session_state.last_ticket = result.get("ticket_id", "demo-ticket")
        st.session_state.messages.append({
            "role":      "assistant",
            "content":   result["answer"],
            "citations": result.get("citations",[]),
            "conflict":  result.get("conflict_explanation",""),
        })
        st.rerun()

# ── Right panel ───────────────────────────────────────────────────────────────
with col_panel:
    result = st.session_state.last_result
    ticket = st.session_state.last_ticket

    st.markdown("#### Sources")
    if result:
        for c in result.get("citations",[]):
            with st.container(border=True):
                st.markdown(f"**{c.get('filename','—')}**")
                st.caption(f"{c.get('section','—')} · {c.get('doc_date','—')}")
                if c.get("excerpt"):
                    st.markdown(
                        f"<blockquote style='font-size:12px'>{c['excerpt'][:140]}</blockquote>",
                        unsafe_allow_html=True)
    else:
        st.caption("Sources appear after your first query.")

    st.divider()
    st.markdown("#### Support ticket")
    if ticket:
        with st.container(border=True):
            st.markdown(f"**Customer:** {customer_name}")
            st.markdown(f"**ID:** `{ticket[:8]}...`")
            st.markdown(f"**Query:** {query[:60] if query and isinstance(query, str) else 'Demo query'}")
            st.markdown(f"**Conflict:** {'Yes ⚠️' if result and result.get('conflict_detected') else 'No ✓'}")
            if result and result.get("conflict_explanation"):
                st.caption(result["conflict_explanation"][:120])
            st.success(f"Saved to crm_tickets/{ticket[:8]}....json ✓")

    st.divider()
    st.markdown("#### Recent tickets")
    st.caption("Demo mode - tickets saved to crm_tickets/ folder")