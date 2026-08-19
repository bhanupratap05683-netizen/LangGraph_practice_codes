from langchain_ollama import ChatOllama

model = ChatOllama(
    model="qwen2.5-coder:7b",
    temperature=0.0,    
    base_url="http://localhost:11434"
)
print("----------Pucho kya puchna hai?----------")
while True:
    prompt = input("You: ")
    if prompt.lower() in ["exit", "quit"]:
        print("Chalo bhaag jao! Bye!")
        break
    response = model.invoke(prompt)
    print(f"Gyani: {response.content}")    