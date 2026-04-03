from storage import get_chroma_client
client = get_chroma_client()
collection = client.get_collection(name='documents')
results = collection.get(include=['metadatas', 'documents'])
print('Number of chunks:', len(results['documents']))
if results['metadatas']:
    print('Sample metadata:', results['metadatas'][0])
    print('Sample content:', results['documents'][0][:200])