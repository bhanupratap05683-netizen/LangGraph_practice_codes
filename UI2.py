import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# --- Page Configuration ---
st.set_page_config(page_title="Gyani AI", page_icon="🧠", layout="centered")

# --- Initialize Session State ---
# Streamlit reruns the script on every click, so we use session_state to persist data
if "messages" not in st.session_state:
    st.session_state.messages = []  # Stores only Human and AI messages

if "current_mode" not in st.session_state:
    st.session_state.current_mode = "Funny"

# --- Load LLM Model (Cached for performance) ---
@st.cache_resource
def load_model():
    return ChatOllama(
        model="qwen2.5-coder:7b",
        temperature=0.0,    
        base_url="http://localhost:11434"
    )

model = load_model()

# --- Define Modes ---
mode_prompts = {
    "Funny": "You are a funny AI agent named Gyani. Respond in a highly humorous, witty, and funny way. Use jokes and puns.",
    "Angry": "You are an angry AI agent named Gyani. Respond very aggressively, impatiently, and rudely. Act like you are annoyed that the user is bothering you.",
    "Sad": "You are a very sad and depressed AI agent named Gyani. Respond in a melancholic, sorrowful, and emotional way. Sigh often and act heartbroken."
}

# --- Sidebar UI ---
with st.sidebar:
    st.title("🧠 Gyani AI")
    st.markdown("### Aapka swagat hai!")
    st.markdown("Choose the mood of your AI:")
    
    # Dropdown for mode selection
    selected_mode = st.selectbox(
        "Select Mode", 
        options=list(mode_prompts.keys()),
        index=list(mode_prompts.keys()).index(st.session_state.current_mode)
    )
    
    # Update mode if changed
    if selected_mode != st.session_state.current_mode:
        st.session_state.current_mode = selected_mode
        st.rerun() # Rerun to update the UI title immediately

    st.divider()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- Main Chat UI ---
st.title(f"Gyani (Mode: {st.session_state.current_mode})")
st.caption("Pucho kya puchna hai?")

# Display chat history from session state
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Chat input
if prompt := st.chat_input("Apna sawaal yahan likhein..."):
    # 1. Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 2. Add user message to history
    st.session_state.messages.append(HumanMessage(content=prompt))
    
    # 3. Prepare context for the model (System Prompt + Chat History)
    system_msg = SystemMessage(content=mode_prompts[st.session_state.current_mode])
    chat_history = [system_msg] + st.session_state.messages
    
    # 4. Generate and display AI response
    with st.chat_message("assistant"):
        with st.spinner("Gyani soch raha hai..."):
            response = model.invoke(chat_history)
            st.markdown(response.content)
    
    # 5. Add AI response to history
    st.session_state.messages.append(AIMessage(content=response.content))