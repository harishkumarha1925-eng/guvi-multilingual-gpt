import streamlit as st
from src.config import settings

def handle_turn():
    st.title("Multilingual GPT (stub)")
    # replay chat history
    if "history" not in st.session_state:
        st.session_state.history = []
    for role, content in st.session_state.history:
        with st.chat_message(role):
            st.markdown(content)

    # input box
    user_msg = st.chat_input("Say something…")
    if user_msg:
        st.session_state.history.append(("user", user_msg))
        with st.chat_message("user"):
            st.markdown(user_msg)

        # dummy reply so the app runs
        reply = f"[{settings.device}] You said: {user_msg}"
        st.session_state.history.append(("assistant", reply))
        with st.chat_message("assistant"):
            st.markdown(reply)
