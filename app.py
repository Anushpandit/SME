import streamlit as st
import os
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from ingestion import ingest_file, infer_source_type, extract_dates_from_text
from storage import get_chroma_client, store_document
from retrieval import retrieve_relevant_chunks, retrieve_all_chunks
from reasoning import resolve_conflicts_and_reason, classify_intent, find_supporting_snippets


def process_upload_file(client, uploaded_file):
    file_path = f"./temp_{uploaded_file.name}"
    try:
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        markdown = ingest_file(file_path)
        source_type = infer_source_type(file_path)
        file_date = None
        if isinstance(markdown, str):
            file_date = extract_dates_from_text(markdown)
        elif isinstance(markdown, dict) and 'file_date' in markdown:
            file_date = markdown['file_date']

        source_name = uploaded_file.name
        upload_date = datetime.now()
        store_document(client, source_name, markdown, upload_date, source_type=source_type, file_date=file_date)

        return (source_name, True, f"{source_name} ingested")
    except Exception as e:
        return (uploaded_file.name, False, str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


st.title("Local RAG Application for SMEs")

st.markdown("Drag and drop files onto the uploader or click to browse.\n\nSupports PDFs, Excel, email/text files and images.")

# File uploader (streamlit built-in drag/drop behavior)
uploaded_files = st.file_uploader("Upload PDFs, Excel, or email/text files", accept_multiple_files=True, type=['pdf', 'xlsx', 'xls', 'eml', 'txt', 'png', 'jpg', 'jpeg'], key='file_uploader')

# Clear all stored data button
if st.button("Clear stored data"):
    client = get_chroma_client()
    collection = client.get_collection(name='documents')

    try:
        # Get all IDs in collection and delete by IDs.
        current = collection.get(include=['metadatas'])
        metadatas = current.get('metadatas', [])
        # each metadata item is a dict with chunk_id
        all_ids = [m.get('chunk_id') for m in metadatas if isinstance(m, dict) and 'chunk_id' in m]
        if all_ids:
            collection.delete(ids=all_ids)
        else:
            # no docs to delete
            st.info("No documents to delete.")
    except Exception as e:
        st.error(f"Failed to clear stored data: {e}")
    else:
        st.success("All stored data cleared.")

if uploaded_files:
    client = get_chroma_client()
    st.info("Ingestion running in the background. The app remains responsive.")
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(process_upload_file, client, f): f for f in uploaded_files}
        for future in as_completed(futures):
            uploaded_file = futures[future]
            try:
                source_name, ok, msg = future.result()
                if ok:
                    st.success(f"{source_name} ingested and stored successfully!")
                else:
                    st.error(f"{source_name} ingestion failed: {msg}")
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

show_reasoning = st.checkbox("View reasoning details", value=False, help="Show logic steps, conflict analysis, and source scores.")

if prompt := st.chat_input("Ask a question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    try:
        # Small-talk / broad query guard
        ql = prompt.strip().lower()
        if len(ql) < 40 and any(w in ql for w in ['tell me about', 'overview', 'can you explain', 'info on']):
            st.warning("Your question seems broad. Can you specify product or policy area (e.g., Widget A, refund policy, pricing table)?")
            answer = "Broad query received; please clarify the specific product, policy, or period you need."
        else:
            # Route by intent to reduce noise
            intent = classify_intent(prompt)
            source_filter = None
            if intent == 'Pricing':
                source_filter = ['policy', 'table']
            elif intent == 'Policy':
                source_filter = ['policy', 'email', 'table']
            elif intent == 'Communication':
                source_filter = ['email']

            # If a communication query also includes policy/refund terms, broaden to policy/email/table
            if intent == 'Communication' and any(k in ql for k in ['refund', 'return', 'cancel', 'policy', 'warranty', 'price', 'pricing']):
                source_filter = ['email', 'policy', 'table']

            # Date normalization: if query includes year/month use as date filter
            date_after = None
            date_before = None
            import re
            date_match = re.search(r"(\d{4}-\d{2}-\d{2})", prompt)
            if date_match:
                date_after = datetime.fromisoformat(date_match.group(1))

            # Check if user is asking about a specific file
            filename_match = re.search(r'(\w+\.(?:pdf|eml|txt|xlsx|xls|jpg|jpeg|png))', prompt, re.IGNORECASE)
            file_query = False
            
            if not filename_match:
                # Also try to match filename without extension
                filename_match = re.search(r"(?:content of|what(?:'s|s| is)?(?:\s+the)?\s+(?:content of\s+)?in|show me|tell me about|read)\s+['\"`]?(\w+)['\"`]?", prompt, re.IGNORECASE)
                if filename_match:
                    file_query = True
            else:
                file_query = True
            
            if file_query and filename_match:
                target_name = filename_match.group(1).lower()
                all_chunks = retrieve_all_chunks()
                # Match files by name (with or without extension)
                chunks = [c for c in all_chunks if c['source_name'].lower().startswith(target_name) or c['source_name'].lower().split('.')[0] == target_name]
                if chunks:
                    st.info(f"Retrieved content for {chunks[0]['source_name']}")
                else:
                    st.warning(f"File '{target_name}' not found in database")
            else:
                chunks = retrieve_relevant_chunks(prompt, source_types=source_filter, date_after=date_after, date_before=date_before)

            if not chunks and source_filter:
                # fallback to broad search if intent filtering misses content
                chunks = retrieve_relevant_chunks(prompt, n_results=10)
                if chunks:
                    st.warning("No intent-filtered results; using broader search results.")

            # Additional fallback for email metadata queries
            if not chunks and any(k in ql for k in ['who is sender', 'sender', 'subject', 'date', 'body', 'email body', 'content of email', 'from:']):
                chunks = retrieve_relevant_chunks(prompt, source_types=['email'], n_results=15)
                if not chunks:
                    all_chunks = retrieve_all_chunks()
                    chunks = [c for c in all_chunks if c.get('source_type') == 'email' or 'from:' in c.get('content', '').lower()]
                if chunks:
                    st.warning("Email metadata query fallback used.")

            # If querying for a specific product, prioritize exact matching clauses
            if 'widget' in ql or 'product' in ql:
                exact_target = None
                for p in ['widget a', 'widget b', 'widget c']:
                    if p in ql:
                        exact_target = p
                        break
                if exact_target:
                    exact = [c for c in chunks if exact_target in c['content'].lower()]
                    if exact:
                        chunks = exact

            # show retrieved docs for this query to avoid confusion about stale data
            if chunks:
                st.info("Retrieved sources: " + ", ".join([f"{c['source_name']}[{c.get('chunk_id','N/A')} ]" for c in chunks]))
                answer = resolve_conflicts_and_reason(chunks, prompt)

                # Add a recommended action template for SMEs
                if st.button("Draft written request (email / ticket)"):
                    draft = f"Dear Support Team,\n\nI need assistance with: {prompt}\n\nCurrent policy context: {chunks[0]['content'][:250]}...\n\nPlease advise next steps.\n\nRegards," 
                    st.text_area("Draft (editable)", value=draft, height=180)
            else:
                answer = f"No record found for '{prompt}' in current database. No hallucinated content will be returned."

        if not show_reasoning and 'CRITIC VERDICT' in answer:
            # hide the verbose part if user doesn't want reasoning details
            answer = answer.split('CRITIC VERDICT:')[0].strip()

    except Exception as e:
        answer = f"Error generating answer: {str(e)}. Please check Ollama setup."
    
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

# Generate Support Ticket
if st.button("Generate Support Ticket"):
    if st.session_state.messages:
        last_query = next((m["content"] for m in reversed(st.session_state.messages) if m["role"] == "user"), "")
        last_answer = next((m["content"] for m in reversed(st.session_state.messages) if m["role"] == "assistant"), "")
        sources = retrieve_relevant_chunks(last_query)
        
        ticket = {
            "ticket": {
                "query": last_query,
                "answer": last_answer,
                "sources": [{"source_name": s["source_name"], "upload_date": s["upload_date"], "content": s["content"]} for s in sources]
            }
        }

        # Log support ticket to file
        with open("support_ticket_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(ticket, ensure_ascii=False) + "\n")

        st.success("Support ticket saved to support_ticket_log.jsonl")

        # Show mock CRM form
        st.subheader("Mock CRM Support Ticket Form")
        with st.form("crm_form"):
            st.text_input("Customer Query", value=last_query, disabled=True)
            st.text_area("AI Response", value=last_answer, height=100, disabled=True)
            sources_text = "\n".join([f"{s['source_name']} ({s['upload_date']})" for s in sources])
            st.text_area("Sources", value=sources_text, height=50, disabled=True)
            st.text_input("Priority", value="Medium")
            st.text_input("Assignee", value="Support Team")
            submitted = st.form_submit_button("Submit Ticket")
            if submitted:
                st.success("Ticket submitted to CRM!")
        
        # Download JSON
        json_str = json.dumps(ticket, indent=4)
        st.download_button("Download Ticket JSON", json_str, "support_ticket.json", "application/json")
    else:
        st.error("No chat history to generate ticket from.")