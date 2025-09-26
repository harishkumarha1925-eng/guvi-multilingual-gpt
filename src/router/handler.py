import streamlit as st

def handle_turn() -> None:
    st.title("Multilingual GPT (stub)")

    # simple chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # render history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # input + echo reply
    user_text = st.chat_input("Say something...")
    if user_text:
        st.session_state["messages"].append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.write(user_text)

        reply = "Echo: " + user_text
        st.session_state["messages"].append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.write(reply)

