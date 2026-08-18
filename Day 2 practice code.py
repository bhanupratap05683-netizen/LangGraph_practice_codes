from langchain_ollama import OllamaEmbeddings

# 1. Initialize the embedding model
embeddings = OllamaEmbeddings(
    model="qwen2.5-coder:7b",
    base_url="http://localhost:11434"  # Default Ollama local endpoint
)

# 2. Generate an embedding for a single query/string
single_text = "Write a Python function to compute the Fibonacci sequence."
single_vector = embeddings.embed_query(single_text)

print(f"Single vector dimensions: {len(single_vector)}")
print(f"First 5 values: {single_vector[:5]}\n")

# 3. Generate embeddings for a batch of documents
documents = [
    "def bubble_sort(arr): ...",
    "def quick_sort(arr): ...",
    "SELECT * FROM users WHERE active = 1;",
    "CSS Grid vs Flexbox layout comparison."
]

doc_vectors = embeddings.embed_documents(documents)

print(f"Generated {len(doc_vectors)} document embeddings.")
print(f"Document 1 vector dimensions: {len(doc_vectors[0])}")