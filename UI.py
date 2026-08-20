from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import streamlit as st

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gyani - Funny AI",
    page_icon="🤡",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
        .main {
            background-color: #0e1117;
        }

        .title-container {
            text-align: center;
            padding: 20px 0 10px 0;
        }

        .title-container h1 {
            font-size: 3rem;
            color: #f0a500;
            font-weight: 800;
            letter-spacing: 2px;
        }

        .title-container p {
            color: #aaaaaa;
            font-size: 1rem;
            margin-top: -10px;
        }

        .chat-container {
            max-height: 520px;
            overflow-y: auto;
            padding: 10px 5px;
            margin-bottom: 10px;
        }

        .user-bubble {
            background-color: #1f6feb;
            color: white;
            padding: 12px 16px;
            border-radius: 18px 18px 4px 18px;
            margin: 8px 0;
            max-width: 75%;
            margin-left: auto;
            font-size: 0.95rem;
            line-height: 1.5;
            word-wrap: break-word;
        }

        .ai-bubble {
            background-color: #21262d;
            color: #e6edf3;
            padding: 12px 16px;
            border-radius: 18px 18px 18px 4px;
            margin: 8px 0;
            max-width: 75%;
            margin-right: auto;
            font-size: 0.95rem;
            line-height: 1.5;
            word-wrap: break-word;
            border-left: 4px solid #f0a500;
        }

        .avatar-user {
            text-align: right;
            font-size: 0.75rem;
            color: #8b949e;
            margin-bottom: 2px;
        }

        .avatar-ai {
            text-align: left;
            font-size: 0.75rem;
            color: #8b949e;
            margin-bottom: 2px;
        }

        .divider {
            border: none;
            border-top: 1px solid #21262d;
            margin: 10px 0;
        }

        .stTextInput > div > div > input {
            background-color: #161b22;
            color: #e6edf3;
            border: 1px solid #30363d;
            border-radius: 10px;
            padding: 12px 15px;
            font-size: 1rem;
        }

        .stButton > button {
            background-color: #f0a500;
            color: #0e1117;
            border: none;
            border-radius: 10px;
            padding: 10px 24px;
            font-weight: 700;
            font-size: 1rem;
            transition: 0.3s;
        }

        .stButton > button:hover {
            background-color: #d4920a;
            color: white;
        }

        .clear-btn > button {
            background-color: #21262d !important;
            color: #e6edf3 !important;
            border: 1px solid #30363d !important;
        }

        .clear-btn > button:hover {
            background-color: #da3633 !important;
            color: white !important;
        }

        .status-bar {
            background-color: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 8px 14px;
            font-size: 0.8rem;
            color: #8b949e;
            margin-bottom: 12px;
            display: flex;
            justify-content: space-between;
        }

        .thinking-text {
            color: #f0a500;
            font-style: italic;
            font-size: 0.9rem;
            text-align: center;
            padding: 8px;
        }
    </style>
""", unsafe_allow_html=True)


# ── Model Init ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return ChatOllama(
        model="qwen2.5-coder:7b",
        temperature=0.0,
        base_url="http://localhost:11434"
    )

model = load_model()


# ── Session State ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a Agent that response in funny way.")
    ]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role": ..., "content": ...}


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
    <div class="title-container">
        <h1>🤡 Gyani AI</h1>
        <p>Pucho kya puchna hai? — Ask anything, get funny answers!</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)


# ── Status Bar ────────────────────────────────────────────────────────────────
msg_count = len(st.session_state.chat_history)
st.markdown(f"""
    <div class="status-bar">
        <span>🟢 Model: <b>qwen2.5-coder:7b</b></span>
        <span>💬 Messages: <b>{msg_count}</b></span>
        <span>🌡️ Temperature: <b>0.0</b></span>
    </div>
""", unsafe_allow_html=True)


# ── Chat Display ──────────────────────────────────────────────────────────────
chat_placeholder = st.container()

with chat_placeholder:
    if not st.session_state.chat_history:
        st.markdown("""
            <div style='text-align:center; color:#444d56; padding: 60px 0;'>
                <div style='font-size: 3rem;'>🤡</div>
                <div style='font-size: 1rem; margin-top: 10px;'>
                    Say something! Gyani is waiting to roast... uhh help you!
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        for chat in st.session_state.chat_history:
            if chat["role"] == "user":
                st.markdown(f"""
                    <div class="avatar-user">You 👤</div>
                    <div class="user-bubble">{chat["content"]}</div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="avatar-ai">🤡 Gyani</div>
                    <div class="ai-bubble">{chat["content"]}</div>
                """, unsafe_allow_html=True)


# ── Input Area ────────────────────────────────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([7, 1.5, 1.5])

with col1:
    user_input = st.text_input(
        label="input",
        placeholder="Type your message here...",
        label_visibility="collapsed",
        key="user_input"
    )

with col2:
    send_clicked = st.button("Send 🚀", use_container_width=True)

with col3:
    st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
    clear_clicked = st.button("Clear 🗑️", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ── Clear Chat ────────────────────────────────────────────────────────────────
if clear_clicked:
    st.session_state.messages = [
        SystemMessage(content="You are a Agent that response in funny way.")
    ]
    st.session_state.chat_history = []
    st.rerun()


# ── Handle Send ───────────────────────────────────────────────────────────────
if send_clicked and user_input.strip():
    prompt = user_input.strip()

    # Append to langchain messages
    st.session_state.messages.append(HumanMessage(content=prompt))

    # Append to display history
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # Show thinking indicator
    with st.spinner("🤡 Gyani is cooking something funny..."):
        response = model.invoke(st.session_state.messages)

    # Append AI response
    st.session_state.messages.append(AIMessage(content=response.content))
    st.session_state.chat_history.append({"role": "ai", "content": response.content})

    st.rerun()


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
    <div style='text-align: center; color: #444d56; font-size: 0.75rem; padding: 20px 0 5px 0;'>
        Powered by <b>Ollama</b> + <b>LangChain</b> + <b>Streamlit</b> 🚀
    </div>
""", unsafe_allow_html=True)