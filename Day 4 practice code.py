from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
# 1. Initialize the loader with the correct path
loader = PyPDFLoader("GRU.pdf")

# 2. Load the PDF pages into a list
docs = loader.load()

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that provides summaries of documents based on the provided document."),("human", "{loader}")])

model = ChatOllama(
    model="qwen2.5-coder:7b",
    temperature=0.0,    
    base_url="http://localhost:11434"
)
prompt = template.format_prompt(loader=docs[0].page_content).to_messages()
result = model.invoke(prompt)  
print(result.content)