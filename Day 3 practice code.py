from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

model = ChatOllama(
    model="qwen2.5-coder:7b",
    temperature=0.0,    
    base_url="http://localhost:11434"
)
messages = [ SystemMessage(content="You are a Agent that response in funny way."),
]
print("----------Pucho kya puchna hai?----------")
while True:
    prompt = input("You: ")
    messages.append(HumanMessage(content=prompt))
    if prompt.lower() in ["exit", "quit"]:
        print(messages)
        print("Chalo bhaag jao! Bye!")
        break
    response = model.invoke(messages)
    messages.append(AIMessage(content=response.content))
    print(f"Gyani: {response.content}")    