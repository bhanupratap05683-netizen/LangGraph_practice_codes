from langchain_community.document_loaders import PyPDFLoader

# 1. Initialize the loader with the correct path
loader = PyPDFLoader("GRU.pdf")

# 2. Load the PDF pages into a list
docs = loader.load()

# 3. Access the first page's content
print(docs[0].page_content)