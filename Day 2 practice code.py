from langchain_ollama import OllamaEmbeddings

# 1. Initialize with a dedicated embedding model
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

# 2. Embed a single query
query_text = "How do I implement quicksort in Python?"
query_vector = embeddings.embed_query(query_text)

print(f"Query Vector Dimensions: {len(query_vector)}")
print(f"Sample values: {query_vector[:5]}\n")

# 3. Embed a list of documents/code snippets
documents = [
    "def quicksort(arr): return arr if len(arr) <= 1 else quicksort([x for x in arr[1:] if x < arr[0]]) + [arr[0]] + quicksort([x for x in arr[1:] if x >= arr[0]])",
    "def bubble_sort(arr): ...",
    "SELECT * FROM users WHERE status = 'active';",
    "const fetchData = async () => await fetch('/api/data');"
]

doc_vectors = embeddings.embed_documents(documents)

print(f"Total documents embedded: {len(doc_vectors)}")
print(f"Document 1 vector dimensions: {len(doc_vectors[0])}")