from ingestion import ingest_file
from storage import store_document, get_chroma_client
from datetime import datetime

md = ingest_file('c:/Users/Abhay/Downloads/sample-email.eml')
client = get_chroma_client()
store_document(client, 'sample-email.eml', md, datetime.now(), source_type='email')
print('Ingested sample email')