import streamlit as st

def init_memory():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

def save_message(question, answer):
    st.session_state.chat_history.append(
        {"question": question, "answer": answer}
    )
