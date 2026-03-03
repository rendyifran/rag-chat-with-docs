# app.py
import time
import streamlit as st

from query_data import load_db, answer_question

st.set_page_config(page_title="🤖 RAG Project Assistant", page_icon="🤖", layout="centered")

st.title("🤖 RAG Project Assistant")
st.caption(
    "This chatbot answers questions about my MSc and applied data science projects "
    "using a custom Retrieval-Augmented Generation (RAG) pipeline."
)

st.markdown(
    "Built with: Chroma (vector DB), Ollama (LLM), custom intent-aware retrieval.\n"
    "GitHub: https://github.com/rendyifran"
)

# Cache DB load so it happens only once per app run
@st.cache_resource
def get_db():
    return load_db()

db = get_db()

# Session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of dicts: {role, content, sources?}

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    if st.button("🧹 Clear chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("Display")
    show_sources = st.checkbox("Show sources", value=True)

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if show_sources and msg["role"] == "assistant":
            sources = msg.get("sources") or []
            if sources:
                st.caption("Sources: " + ", ".join(sources))

# Input
user_query = st.chat_input("Type your question…")

if user_query:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            start = time.time()
            answer, sources, intent = answer_question(db, user_query)
            elapsed = time.time() - start

        st.write(answer)

        if show_sources and sources:
            st.caption("Sources: " + ", ".join(sources))

        st.caption(f"Intent: {intent} • {elapsed:.2f}s")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources, "intent": intent}
    )