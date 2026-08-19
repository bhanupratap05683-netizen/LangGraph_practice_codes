import os
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver

llm = init_chat_model(
    model="qwen2.5-coder:7b",
    temperature=0.0,
    model_provider="ollama",
    base_url="http://localhost:11434"
)
response = llm.invoke("what is the capital of france?")
print(response)