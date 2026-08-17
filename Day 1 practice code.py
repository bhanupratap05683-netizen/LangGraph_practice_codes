import os
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
llm = init_chat_model(
    model="qwen2.5-coder:7b",
    model_provider="ollama",
    temperature=0.0,
    base_url="http://localhost:11434"
)
response = llm.invoke("Write a Python function that takes a list of numbers and returns the sum of the even numbers in the list.")
print(response)