from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

model = ChatOllama(
    model="qwen2.5-coder:7b",
    temperature=0.0,    
    base_url="http://localhost:11434"
)

print("----------Gyani aapka swagat karta hai!----------")

print("1 dail kare funny response ke liye")
print("2 dail kare Angry response ke liye")
print("3 dail kare sad response ke liye")
choice = int(input("Aapka choice kya hai? (1/2/3): "))

if choice == 1:
    mode = "you are a funny AI agent response in a funny way"
elif choice == 2:
    mode = "you are an angry AI agent, response very aggresivley and impationttly"
elif choice == 3:
    mode = "you are a sad AI agent response in a sad way"  

messages = [ SystemMessage(content=mode),
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