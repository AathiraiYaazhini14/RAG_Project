from ingestion import ingest_document

with open('test.txt', 'rb') as f:
    content = f.read()

result = ingest_document(content, 'test.txt', 'text/plain')
print(result)