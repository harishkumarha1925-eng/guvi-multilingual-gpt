# app.py  — minimal, TypeError-proof Streamlit app
import streamlit as st
from src.router import handle_turn  # must return a string

st.set_page_config(page_title="GUVI Multilingual GPT Chatbot", page_icon="🤖")

# --- sidebar --------------------------------------------------------------
st.sidebar.header("Settings")
domain_mode = st.sidebar.selectbox(
    "Domain mode", ["general", "technical", "educational", "friendly"], index=0
)

with st.sidebar.expander("Examples", expanded=False):
    st.markdown(
        "- general: *What's the capital of Japan?*\n"
        "- technical: *Explain Python generators with a short code sample.*\n"
        "- educational: *Teach fractions to a 10-year-old with an example.*\n"
        "- friendly: *Write a cheerful 2-line greeting.*"
    )

# --- header ---------------------------------------------------------------
st.title("GUVI Multilingual GPT Chatbot")
st.caption("LLM Mode: `local_small`  •  Status: ready")

# --- chat state -----------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list[dict(role, content)]

# render history
for m in st.session_state.messages:
    st.chat_message(m["role"]).write(m["content"])

# --- input ---------------------------------------------------------------
user_text = st.chat_input("Type your message in any language…")

def _safe_str(x) -> str:
    return "" if x is None else str(x)

if user_text:
    # show user message
    user_text = _safe_str(user_text).strip()
    st.session_state.messages.append({"role": "user", "content": user_text})
    st.chat_message("user").write(user_text)

    # get assistant reply safely
    try:
        with st.spinner("Thinking & translating…"):
            reply = handle_turn(user_text, domain_role=_safe_str(domain_mode))
        reply = _safe_str(reply).strip()
        if not reply:
            reply = "⚠️ Sorry—no response."
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.chat_message("assistant").write(reply)
    except Exception as e:
        # show the actual exception type so we can diagnose
        import traceback
        st.toast("Something went wrong while generating the answer.", icon="⚠️")
        st.error(f"Internal error: {type(e).__name__}")
        st.code(traceback.format_exc())

