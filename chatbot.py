from dotenv import load_dotenv
import os
import json
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory

# =======================
# Load environment
# =======================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# =======================
# Streamlit config
# =======================
st.set_page_config(page_title="🧠 BrainWave", layout="centered")
st.title("Welcome to 🧠 BrainWave")

# =======================
# Sidebar
# =======================
with st.sidebar:
    st.subheader("Control & Tools 🛠️")

    model_name = st.selectbox(
        "Choose model",
        [
            "llama-3.1-8b-instant",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-safeguard-20b",
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "gpt-4o-mini",
            "gpt-4o",
            "llama-3.1-70b",
            "mixtral-8x7b"
        ],
        index=1
    )

    temperature = st.slider("Creativity", 0.0, 1.0, 0.7)
    max_tokens = st.slider("Tokens", 100, 500, 150)

    system_prompt = st.text_area(
        "System Prompt",
        value="Give short answers. Be sweet and friendly."
    )

    if st.button("🧹 Remove Chat"):
        st.session_state.pop("history", None)
        st.rerun()

# =======================
# API key check
# =======================
if not GROQ_API_KEY:
    st.error("🔑 GROQ API key missing. Add it to your .env file.")
    st.stop()

# =======================
# Chat history
# =======================
if "history" not in st.session_state:
    st.session_state.history = InMemoryChatMessageHistory()

# =======================
# LLM
# =======================
llm = ChatGroq(
    model=model_name,
    temperature=temperature,
    max_tokens=max_tokens
)

# =======================
# Prompt
# =======================
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# =======================
# Chain
# =======================
chain = prompt | llm | StrOutputParser()

with_message_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: st.session_state.history,
    input_messages_key="input",
    history_messages_key="history",
)

# =======================
# Display chat history
# =======================
for msg in st.session_state.history.messages:
    if msg.type == "human":
        st.chat_message("user").write(msg.content)
    else:
        st.chat_message("assistant").write(msg.content)

# =======================
# User input
# =======================
user_input = st.chat_input("Ask anything...")

if user_input:
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        try:
            response = with_message_history.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": "default"}}
            )
            st.write(response)

        except Exception as e:
            st.error(f"Model error: {e}")

# =======================
# Export chat history
# =======================
export = []

for m in st.session_state.history.messages:
    if m.type == "human":
        export.append({"role": "user", "content": m.content})
    elif m.type in ("ai", "assistant"):
        export.append({"role": "assistant", "content": m.content})

if export:
    json_data = json.dumps(export, ensure_ascii=False, indent=2)

    st.download_button(
        label="⬇️ Download chat JSON",
        data=json_data,
        file_name="chat_history.json",
        mime="application/json",
        use_container_width=True
    )
