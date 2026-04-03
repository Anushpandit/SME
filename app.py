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

# API Key Configuration
st.sidebar.header("🔑 API Configuration")
st.sidebar.markdown("**Required for AI responses**")
api_key_input = st.sidebar.text_input("Enter Groq API Key", type="password",
                                   help="Get from https://console.groq.com/",
                                   placeholder="gsk_...")

if api_key_input:
    st.session_state["groq_api_key"] = api_key_input
    st.sidebar.success("✅ API Key saved to session")
elif "groq_api_key" in st.session_state:
    st.sidebar.info("✅ API Key loaded from session")
else:
    st.sidebar.error("❌ Please enter your Groq API Key to enable AI responses")
    st.sidebar.markdown("**Without API key, responses will use basic text matching only**")

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
    
    # Check for duplicate files before processing
    collection = client.get_collection(name='documents')
    existing_files = set()
    try:
        current = collection.get(include=['metadatas'])
        metadatas = current.get('metadatas', [])
        existing_files = set(m.get('source_name', '') for m in metadatas if isinstance(m, dict))
    except:
        pass  # Collection might not exist yet
    
    # Filter out duplicates
    files_to_process = [f for f in uploaded_files if f.name not in existing_files]
    duplicate_files = [f.name for f in uploaded_files if f.name in existing_files]
    
    if duplicate_files:
        st.warning(f"Skipped duplicate files: {', '.join(duplicate_files)}")
    
    if files_to_process:
        st.info("Ingestion running in the background. The app remains responsive.")
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(process_upload_file, client, f): f for f in files_to_process}
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

# Check if API key is available
api_key_available = (
    "groq_api_key" in st.session_state or 
    (hasattr(st, 'secrets') and "GROQ_API_KEY" in st.secrets) or
    os.getenv('GROQ_API_KEY')
)

if not api_key_available:
    st.warning("⚠️ Please enter your Groq API Key in the sidebar to enable AI responses.")
    st.info("Get your free API key from: https://console.groq.com/")
    st.stop()  # Stop execution until API key is provided

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

show_reasoning = st.checkbox("View Reasoning", value=False, help="Show raw retrieved chunks and their metadata (Source ID and Date).")

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
                answer = resolve_conflicts_and_reason(chunks, prompt, stream=False)

                # Display reasoning details if requested
                if show_reasoning:
                    with st.expander("Retrieved Chunks and Metadata", expanded=True):
                        for i, chunk in enumerate(chunks, 1):
                            st.markdown(f"**Chunk {i}: {chunk['source_name']}**")
                            st.markdown(f"- **Source ID**: {chunk.get('chunk_id', 'N/A')}")
                            st.markdown(f"- **Date**: {chunk.get('file_date', 'N/A')}")
                            st.markdown(f"- **Upload Date**: {chunk['upload_date']}")
                            st.markdown(f"- **Source Type**: {chunk.get('source_type', 'N/A')}")
                            st.text_area(f"Content {i}", value=chunk['content'], height=100, disabled=True, key=f"chunk_{i}")
                            st.markdown("---")

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
        if isinstance(answer, str):
            st.markdown(answer)
        elif hasattr(answer, "__iter__"):
            # Fallback generator streaming implementation
            output = st.empty()
            display_text = ""
            for chunk in answer:
                display_text += chunk
                output.markdown(display_text)
        else:
            try:
                st.markdown(str(answer))
            except Exception:
                st.write(answer)
    
    # Generate Support Ticket button - only show after successful response
    if "messages" in st.session_state and len(st.session_state.messages) >= 2:
        last_user_msg = None
        last_assistant_msg = None
        for msg in reversed(st.session_state.messages):
            if msg["role"] == "assistant" and not last_assistant_msg:
                last_assistant_msg = msg["content"]
            elif msg["role"] == "user" and not last_user_msg:
                last_user_msg = msg["content"]
            if last_user_msg and last_assistant_msg:
                break
        
        if last_user_msg and last_assistant_msg and not last_assistant_msg.startswith("Error"):
            if st.button("Generate Support Ticket"):
                sources = retrieve_relevant_chunks(last_user_msg)
                
                ticket = {
                    "ticket": {
                        "query": last_user_msg,
                        "answer": last_assistant_msg,
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
                    st.text_input("Customer Query", value=last_user_msg, disabled=True)
                    st.text_area("AI Response", value=last_assistant_msg, height=100, disabled=True)
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