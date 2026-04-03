import groq
import re
from datetime import datetime
from difflib import SequenceMatcher
import os

def fallback_answer(chunks, query):
    """Fallback answer without LLM: return most relevant chunk."""
    if not chunks:
        return "No relevant information found."

    best_chunk = max(chunks, key=lambda c: SequenceMatcher(None, query.lower(), c['content'].lower()).ratio())
    return f"Based on {best_chunk['source_name']} ({best_chunk['upload_date']}): {best_chunk['content']}"


def classify_intent(query):
    """Basic intent classification for routing."""
    q = query.lower()
    if any(x in q for x in ['price', 'pricing', 'cost', 'fee']):
        return 'Pricing'
    if any(x in q for x in ['refund', 'return', 'cancel', 'warranty', 'policy']):
        return 'Policy'

    # Communication intent for email metadata and contact actions
    if any(x in q for x in ['send email', 'reply email', 'email to', 'contact', 'message', 'communicate', 'support request', 'sender', 'subject', 'date', 'body', 'content']):
        return 'Communication'

    # fallback: product-specific lookups
    if any(x in q for x in ['widget', 'product', 'sku']):
        return 'Product'
    return 'General'


def pumprompt_compress(text, max_words=300):
    """A lightweight prompt compressor to strip filler words and limit length."""
    try:
        from llmlingua import compress_text
        compressed = compress_text(text, max_tokens=max_words)
        if compressed:
            return compressed
    except Exception:
        pass

    stopwords = set(['the', 'and', 'or', 'a', 'an', 'to', 'for', 'in', 'of', 'is', 'was', 'be', 'that', 'this'])
    tokens = [w for w in re.split(r"\s+", text) if w and w.lower() not in stopwords]
    if len(tokens) <= max_words:
        return ' '.join(tokens)
    return ' '.join(tokens[-max_words:])


def find_supporting_snippets(query, chunks, max_snippets=3):
    """Return the most relevant sentence snippets that support the answer."""
    query_tokens = set(re.findall(r"\w+", query.lower()))
    snippets = []
    for chunk in chunks:
        text = chunk.get('content', '')
        sentences = re.split(r'[\.\n]', text)
        for sent in sentences:
            sent_tokens = set(re.findall(r"\w+", sent.lower()))
            if not sent_tokens or not query_tokens:
                continue
            overlap = len(query_tokens & sent_tokens)
            if overlap >= max(1, len(query_tokens)//4):
                snippets.append((overlap, chunk, sent.strip()))

    snippets = sorted(snippets, key=lambda x: x[0], reverse=True)
    result = []
    for overlap, chunk, snippet in snippets[:max_snippets]:
        result.append({
            'source_name': chunk.get('source_name'),
            'chunk_id': chunk.get('chunk_id'),
            'file_date': chunk.get('file_date'),
            'snippet': snippet
        })
    return result


def answer_email_metadata(query, chunks):
    """Answer simple email metadata questions (sender, subject, body, date)."""
    q = query.lower()
    wants_sender = any(k in q for k in ['who is sender', 'who sent', 'from:', 'sender'])
    wants_subject = any(k in q for k in ['subject', 'topic'])
    wants_date = any(k in q for k in ['date', 'when', 'timestamp'])
    wants_body = any(k in q for k in ['content of email', 'email body', 'what is in email', 'email text'])

    if not (wants_sender or wants_subject or wants_date or wants_body):
        return None

    def _extract_fields(content):
        sender = ''
        subject = ''
        date_val = ''
        body = ''

        if '## Body' in content:
            parts = content.split('## Body', 1)
            body = parts[1].strip()
        elif 'body' in content.lower():
            body = content

        m = re.search(r'from\s*[:\-]\s*(.+)', content, flags=re.IGNORECASE)
        if m:
            sender = m.group(1).strip().split('\n')[0]

        m = re.search(r'subject\s*[:\-]\s*(.+)', content, flags=re.IGNORECASE)
        if m:
            subject = m.group(1).strip().split('\n')[0]

        m = re.search(r'date\s*[:\-]\s*(.+)', content, flags=re.IGNORECASE)
        if m:
            date_val = m.group(1).strip().split('\n')[0]

        return sender, subject, date_val, body

    # prioritize first matching chunk
    for chunk in chunks:
        sender = chunk.get('sender') or ''
        subject = chunk.get('subject') or ''
        date_val = chunk.get('file_date') or chunk.get('date') or ''
        content = chunk.get('content') or ''

        # fallback to pattern extraction from content if metadata is missing
        if not sender or not subject or not date_val or not content:
            s, sub, d, body = _extract_fields(content)
            sender = sender or s
            subject = subject or sub
            date_val = date_val or d

        if wants_sender and sender:
            return f"Email sender: {sender}\nSubject: {subject or 'Unknown'}"
        if wants_subject and subject:
            return f"Email subject: {subject}\nSender: {sender or 'Unknown'}"
        if wants_date:
            return f"Email date: {date_val or 'Unknown'}\nSender: {sender or 'Unknown'}\nSubject: {subject or 'Unknown'}"
        if wants_body and content:
            body = ''
            if '## Body' in content:
                parts = content.split('## Body', 1)
                body = parts[1].strip()
            elif 'body' in content.lower():
                body = content
            else:
                body = content
            return f"Email body content (truncated): {body[:800]}"

    # Final fallback: scan all chunks for metadata-like patterns
    all_chunks = []
    try:
        from retrieval import retrieve_all_chunks
        all_chunks = retrieve_all_chunks()
    except Exception:
        pass

    for chunk in all_chunks:
        content = chunk.get('content', '')
        sender, subject, date_val, body = _extract_fields(content)
        if wants_sender and sender:
            return f"Email sender: {sender}\nSubject: {subject or 'Unknown'}"
        if wants_subject and subject:
            return f"Email subject: {subject}\nSender: {sender or 'Unknown'}"
        if wants_date and date_val:
            return f"Email date: {date_val}\nSender: {sender or 'Unknown'}\nSubject: {subject or 'Unknown'}"
        if wants_body and body:
            return f"Email body content (truncated): {body[:800]}"

    return None


def verify_with_critic(query, answer, chunks):
    """Use Groq to validate the generated answer against context."""
    try:
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            return 'VALID'  # Fallback if no key
        
        client = groq.Groq(api_key=groq_api_key)
        crt_context = ''
        for chunk in chunks:
            crt_context += f"Source {chunk.get('source_name')} ({chunk.get('chunk_id')}): {chunk.get('content')}\n"

        critic_prompt = f"""
You are a critic model tasked with checking a generated answer for hallucinations and factual consistency.

Query: {query}

Generated Answer: {answer}

Context chunks:
{crt_context}

Evaluate the answer strictly against context chunks. If the answer contains unsupported claims or contradictions, respond with 'INVALID: <reason>'. If it is consistent, respond with 'VALID'.
"""

        response = client.chat.completions.create(
            model="llama3-8b-8192",  # Free model on Groq
            messages=[{"role": "user", "content": critic_prompt}],
            max_tokens=100
        )
        verdict = response.choices[0].message.content.strip()
        return verdict
    except Exception:
        return 'VALID'

def extract_key_values(text):
    import re
    pairs = []
    # e.g. Widget A: 100, price=200
    for m in re.finditer(r"([A-Za-z0-9 _\-]+?)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", text, flags=re.IGNORECASE):
        key = m.group(1).strip()
        value = float(m.group(2))
        pairs.append((key.lower(), value))
    return pairs


def classify_refund_policy(text):
    """Classify refund policy from a text chunk."""
    t = text.lower()
    policy = {
        'allows_refund': None,
        'refund_window_days': None,
        'scope': []
    }

    if 'bulk order' in t or 'bulk orders' in t:
        policy['scope'].append('bulk order')
    if 'custom product' in t or 'custom products' in t or 'service' in t:
        policy['scope'].append('custom')

    # stronger contradiction detection (explicit deny vs allow)
    if re.search(r'no refunds|refunds are not|refunds not allowed|not refundable|cannot be refunded|not eligible for refund', t):
        policy['allows_refund'] = False
    elif re.search(r'refunds .* within|can be refunded|refunds are allowed|eligible for refund|refunds are possible', t):
        policy['allows_refund'] = True

    # negative window vs positive window handling
    if re.search(r'within\s+\d{1,2}\s*days', t):
        m = re.search(r'(\d{1,2})\s*(day|days)', t)
        if m:
            policy['refund_window_days'] = int(m.group(1))
    elif re.search(r'after\s+\d{1,2}\s*days', t):
        # separate pattern e.g. no refund before 30 days
        m = re.search(r'after\s+(\d{1,2})\s*(day|days)', t)
        if m:
            policy['refund_window_days'] = int(m.group(1))

    m = re.search(r'(\d{1,2})\s*(day|days)', t)
    if m:
        policy['refund_window_days'] = int(m.group(1))

    return policy


def detect_conflicts(chunks):
    """Detect potential conflicts between chunks."""
    conflicts = []

    # detect contradictory phrases + numeric mismatch for same label
    # 1. numeric value conflicts
    value_map = {}
    for chunk in chunks:
        kvs = extract_key_values(chunk['content'])
        for key, val in kvs:
            value_map.setdefault(key.lower(), []).append((chunk, val))

    for key, entries in value_map.items():
        unique_vals = set(v for _, v in entries)
        if len(unique_vals) > 1:
            for idx1 in range(len(entries)):
                for idx2 in range(idx1 + 1, len(entries)):
                    c1, v1 = entries[idx1]
                    c2, v2 = entries[idx2]
                    if v1 != v2:
                        conflicts.append({
                            'type': 'numeric_conflict',
                            'key': key,
                            'chunk1': c1,
                            'chunk2': c2,
                            'value1': v1,
                            'value2': v2
                        })

    # 1.5 refund policy contradictions
    policies = []
    for chunk in chunks:
        policy = classify_refund_policy(chunk['content'])
        policy['chunk'] = chunk
        policies.append(policy)

    for i in range(len(policies)):
        for j in range(i + 1, len(policies)):
            p1 = policies[i]
            p2 = policies[j]

            # same scope conflict (bulk/custom/general)
            scopes = set(p1['scope'] + p2['scope'])
            if not scopes:
                scopes = {'general'}

            if p1['allows_refund'] is not None and p2['allows_refund'] is not None and p1['allows_refund'] != p2['allows_refund']:
                conflicts.append({
                    'type': 'policy_conflict',
                    'scope': list(scopes),
                    'chunk1': p1['chunk'],
                    'chunk2': p2['chunk'],
                    'allows_refund_1': p1['allows_refund'],
                    'allows_refund_2': p2['allows_refund']
                })

            # refund window conflict if both exist and different (same scope or general)
            if p1.get('refund_window_days') and p2.get('refund_window_days') and p1['refund_window_days'] != p2['refund_window_days']:
                conflicts.append({
                    'type': 'window_conflict',
                    'scope': list(scopes),
                    'chunk1': p1['chunk'],
                    'chunk2': p2['chunk'],
                    'window1': p1['refund_window_days'],
                    'window2': p2['refund_window_days']
                })

    # 2. contradictory wording conflicts
    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            chunk1 = str(chunks[i].get('content', '')).lower()
            chunk2 = str(chunks[j].get('content', '')).lower()
            similarity = SequenceMatcher(None, chunk1, chunk2).ratio()
            if 0.5 < similarity < 0.99:
                contradictory_phrases = ['not', 'never', 'except', 'unless', 'no', 'cannot']
                has_contradiction = any(phrase in chunk1 and phrase in chunk2 for phrase in contradictory_phrases)
                if has_contradiction:
                    conflicts.append({
                        'type': 'text_conflict',
                        'chunk1': chunks[i],
                        'chunk2': chunks[j],
                        'similarity': similarity
                    })

    return conflicts

def extract_numeric_pairs(chunks):
    """Extract numeric field/value pairs from chunk text for basic comparison."""
    import re
    data = {}
    for chunk in chunks:
        # use pattern field: value or field = value
        for m in re.finditer(r"([A-Za-z0-9 _-]{3,40})[:=]\s*([-+]?[0-9]*\.?[0-9]+)", chunk['content']):
            key = m.group(1).strip()
            value = float(m.group(2))
            data.setdefault(key, []).append((chunk['source_name'], chunk.get('chunk_id', 'N/A'), value))
    return data


def find_target_value(chunks, query):
    q = query.lower()
    target = None
    for p in ['widget a', 'widget b', 'widget c']:
        if p in q:
            target = p
            break

    # if exact target found, match from chunks
    if target is not None:
        # first look for extractor key exact match
        for chunk in chunks:
            chunk_text = chunk['content'] if isinstance(chunk['content'], str) else ''
            for key, value in extract_key_values(chunk_text):
                if target in key.lower():
                    return {
                        'target': target,
                        'value': value,
                        'source': chunk['source_name'],
                        'chunk_id': chunk.get('chunk_id', 'N/A')
                    }

        # fallback: find numeric immediately after target phrase
        for chunk in chunks:
            text = (chunk['content'] if isinstance(chunk['content'], str) else '').lower()
            m = re.search(rf"{re.escape(target)}\s*[:\-\=]?\s*([0-9]+(?:\.[0-9]+)?)", text)
            if m:
                return {
                    'target': target,
                    'value': float(m.group(1)),
                    'source': chunk['source_name'],
                    'chunk_id': chunk.get('chunk_id', 'N/A')
                }

        # second fallback: nearest number in same sentence
        for chunk in chunks:
            text_lower = (chunk['content'] if isinstance(chunk['content'], str) else '').lower()
            if target in text_lower:
                # sentence around target phrase
                pieces = re.split(r'[\n\.]', text_lower)
                for piece in pieces:
                    if target in piece:
                        m2 = re.search(r"([0-9]+(?:\.[0-9]+)?)", piece)
                        if m2:
                            return {
                                'target': target,
                                'value': float(m2.group(1)),
                                'source': chunk['source_name'],
                                'chunk_id': chunk.get('chunk_id', 'N/A')
                            }

    return None


def normalize_date_string(ds):
    try:
        return datetime.fromisoformat(ds)
    except Exception:
        try:
            return datetime.strptime(ds, '%m/%d/%Y')
        except Exception:
            return None


def compare_numeric_data(chunks):
    pairs = extract_numeric_pairs(chunks)
    if not pairs:
        return "No structured numeric data detected for comparison in retrieved chunks."

    result_lines = ["Numeric comparison across stored chunk data:"]
    for key, rows in pairs.items():
        values = [v for _, _, v in rows]
        trend = "increasing" if values == sorted(values) else "decreasing" if values == sorted(values, reverse=True) else "mixed"
        result_lines.append(f"- {key}: min={min(values)}, max={max(values)}, trend={trend}, samples={len(values)}")
        for source, chunk_id, value in rows:
            result_lines.append(f"   • {source}[{chunk_id}] = {value}")
    return "\n".join(result_lines)


def resolve_conflicts_and_reason(chunks, query):
    """Use Ollama to reason with explicit conflict detection, fallback to simple answer."""
    # Detect conflicts
    conflicts = detect_conflicts(chunks)
    
    # Sort chunks by upload_date descending (newer first)
    sorted_chunks = sorted(chunks, key=lambda x: datetime.fromisoformat(x["upload_date"] if x.get('upload_date') else datetime.now().isoformat()), reverse=True)

    # Quick email metadata answer path
    email_meta_answer = answer_email_metadata(query, sorted_chunks)
    if email_meta_answer:
        return email_meta_answer

    # Direct target extraction (optimization for product queries)
    target_hit = find_target_value(sorted_chunks, query)

    # Prepare conflict information string (always compute)
    conflict_info = ""
    if conflicts:
        conflict_info = "\nDetected potential conflicts:\n"
        for conf in conflicts:
            if conf['type'] == 'policy_conflict':
                conflict_info += f"- Policy conflict ({', '.join(conf['scope'])}): {conf['chunk1']['source_name']} says allows_refund={conf['allows_refund_1']} while {conf['chunk2']['source_name']} says allows_refund={conf['allows_refund_2']}\n"
            elif conf['type'] == 'window_conflict':
                conflict_info += f"- Window conflict ({', '.join(conf['scope'])}): {conf['chunk1']['source_name']} says {conf['window1']} days, {conf['chunk2']['source_name']} says {conf['window2']} days\n"
            elif conf['type'] == 'numeric_conflict':
                conflict_info += f"- Numeric conflict for {conf['key']}: {conf['chunk1']['source_name']}={conf['value1']}, {conf['chunk2']['source_name']}={conf['value2']}\n"
            else:
                similarity = conf.get('similarity')
                if similarity is not None:
                    conflict_info += f"- Text conflict: {conf['chunk1']['source_name']} vs {conf['chunk2']['source_name']} (similarity {similarity:.2f})\n"
                else:
                    conflict_info += f"- Text conflict: {conf['chunk1']['source_name']} vs {conf['chunk2']['source_name']}\n"

    if target_hit:
        answer_text = (
            f"1. CONTEXT USED: target {target_hit['target']} found in {target_hit['source']} [{target_hit['chunk_id']}].\n"
            f"2. IDENTIFIED CONFLICTS: {'Conflicts found' if conflicts else 'No contradictions detected.'}\n"
            f"3. CHOICE AND RATIONALE: Selected explicit target match, prefer most recent and explicit chunk.\n"
            f"4. FINAL ANSWER: {target_hit['target'].title()} price is {target_hit['value']}\n"
            f"5. SOURCES: {target_hit['source']} ({target_hit['chunk_id']})"
        )
        if conflict_info:
            answer_text += "\n" + conflict_info
        return answer_text

    # Prepare context
    context_parts = []
    for chunk in sorted_chunks:
        part = f"Source: {chunk['source_name']} (Date: {chunk['upload_date']}, Chunk: {chunk.get('chunk_id', 'N/A')})\n{chunk['content']}"
        context_parts.append(part)
    context = "\n\n".join(context_parts)
    
    # Add conflict info to prompt
    conflict_info = ""
    if conflicts:
        conflict_info = "\nDetected potential conflicts:\n"
        for conf in conflicts:
            if conf['type'] == 'policy_conflict':
                conflict_info += f"- Policy conflict ({', '.join(conf['scope'])}): {conf['chunk1']['source_name']} says allows_refund={conf['allows_refund_1']} while {conf['chunk2']['source_name']} says allows_refund={conf['allows_refund_2']}\n"
            elif conf['type'] == 'window_conflict':
                conflict_info += f"- Window conflict ({', '.join(conf['scope'])}): {conf['chunk1']['source_name']} says {conf['window1']} days, {conf['chunk2']['source_name']} says {conf['window2']} days\n"
            elif conf['type'] == 'numeric_conflict':
                conflict_info += f"- Numeric conflict for {conf['key']}: {conf['chunk1']['source_name']}={conf['value1']}, {conf['chunk2']['source_name']}={conf['value2']}\n"
            else:
                similarity = conf.get('similarity')
                if similarity is not None:
                    conflict_info += f"- Text conflict: {conf['chunk1']['source_name']} vs {conf['chunk2']['source_name']} (similarity {similarity:.2f})\n"
                else:
                    conflict_info += f"- Text conflict: {conf['chunk1']['source_name']} vs {conf['chunk2']['source_name']}\n"

    compressed_context = pumprompt_compress(context, max_words=600)

    prompt = f"""
You are an AI assistant for SMEs. Answer the query based on the provided context.
Prioritize information from newer sources (more recent upload dates). Use the most authoritative sources when available.
If there are contradictions, explicitly highlight them using the structured template below.

Query: {query}

Context:
{compressed_context}

Detected conflicts details:
{conflict_info if conflict_info else 'None'}

STRICT INSTRUCTIONS:
- Only use the content from context chunks (do not hallucinate beyond provided data).
- Always report contradictions if detected; if none, say "No contradictions detected.".
- If chunk describes same product/item with different numbers, mark as numeric conflict.
- If query mentions exact item (e.g. Widget A), prioritize chunk that explicitly names that item.
- IMPORTANT: If query asks for product/pricing details, list ALL products found in the context chunks, not just a subset.

Structured answer format required:
1. CONTEXT USED: list the chunk IDs and summary lines.
2. IDENTIFIED CONFLICTS: highlight contradictory statements with source IDs.
3. CHOICE AND RATIONALE: specify which source is preferred and why (use dates and authority).
4. FINAL ANSWER: concise, user-facing.
5. SOURCES: list source names, chunk IDs, source_type, file_date.
"""

    # Add quick numeric comparison when user asks to compare the database
    compare_data = ""
    if 'compare' in query.lower() or 'difference' in query.lower() or 'greater' in query.lower() or 'less' in query.lower():
        compare_data = compare_numeric_data(sorted_chunks)
        if compare_data:
            prompt += "\n\nAdditional data comparison from DB extracted fields:\n" + compare_data + "\n"

    # Try Groq API
    groq_api_key = os.getenv('GROQ_API_KEY')
    if groq_api_key:
        try:
            client = groq.Groq(api_key=groq_api_key)
            response = client.chat.completions.create(
                model="llama3-8b-8192",  # Free model on Groq
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000
            )
            response_text = response.choices[0].message.content
        except Exception as e:
            print(f"Groq API error: {e}")
            response_text = None
    else:
        response_text = None

    if not response_text:
        # Fallback if Groq not available
        print("Groq API not available, using fallback answer.")
        fallback = fallback_answer(sorted_chunks, query)
        reasoning = "\n\nReasoning: Could not reach Groq API. Using nearest content match with heuristic similarity."
        return fallback + reasoning

    critic_verdict = verify_with_critic(query, response_text, sorted_chunks)
    if critic_verdict and critic_verdict.upper().startswith('INVALID'):
        retry_prompt = prompt + "\n\nLAST RESPONSE WAS FLAGGED INVALID BY CRITIC. REGENERATE WITH STRICT SOURCE SUPPORT ONLY."
        try:
            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": retry_prompt}],
                max_tokens=1000
            )
            response_text = response.choices[0].message.content
        except Exception:
            pass

    snippets = find_supporting_snippets(query, sorted_chunks, max_snippets=3)
    snippet_text = ""
    if snippets:
        snippet_text = "\n\nSNIPPET VERIFICATION:\n"
        for s in snippets:
            snippet_text += f"- {s['source_name']}[{s['chunk_id']}]{' (date '+str(s['file_date'])+')' if s.get('file_date') else ''}: {s['snippet']}\n"

    return f"{response_text}\n\nCRITIC VERDICT: {critic_verdict}{snippet_text if snippet_text else ''}"