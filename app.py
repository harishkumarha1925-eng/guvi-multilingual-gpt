import streamlit as st
from src.config import settings
from src.router import handle_turn
from src.prompts import FAQ_HINT, MENTOR_HINT, RECOMMENDER_HINT
from src.utils import status, toast_ok, toast_warn

st.set_page_config(
    page_title=settings.APP_TITLE,
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Sidebar ----
with st.sidebar:
    st.title("⚙️ Settings")
    mode = st.selectbox("Domain mode", ["general", "faq", "mentor", "recommender"])
    st.caption({
        "general": "Free-form chat.",
        "faq": FAQ_HINT,
        "mentor": MENTOR_HINT,
        "recommender": RECOMMENDER_HINT
    }[mode])
    st.divider()
    st.markdown("**LLM Mode**: " + settings.LLM_MODE)
    if settings.LLM_MODE == "hf_inference":
        st.caption(f"Model: {settings.HF_TEXT_GENERATION_MODEL}")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.pop("history", None)
        toast_ok("Chat cleared")

# ---- Header ----
st.title("GUVI Multilingual GPT Chatbot")
st.caption("Type in any supported language. I’ll auto-translate, reason in English, and reply back in your language.")

# ---- History ----
if "history" not in st.session_state:
    st.session_state["history"] = []  # list of dicts: {"role": "user"/"assistant", "content": str, "meta": dict}

for turn in st.session_state["history"]:
    with st.chat_message(turn["role"]):
        st.write(turn["content"])
        if turn.get("meta") and st.checkbox("Show internals", key=str(hash(turn["content"]))):
            st.json(turn["meta"])

# ---- Input ----
user_msg = st.chat_input("Ask me anything (any language)…")
if user_msg:
    st.session_state["history"].append({"role": "user", "content": user_msg, "meta": {}})
    with st.chat_message("user"):
        st.write(user_msg)

    with st.chat_message("assistant"):
        ph = st.empty()
        with status("Thinking & translating…"):
            try:
                result = handle_turn(user_msg, domain_role=mode)
                ph.write(result["final_answer"])
                st.session_state["history"].append({
                    "role": "assistant", "content": result["final_answer"], "meta": result
                })
            except Exception as e:
                toast_warn("Something went wrong. Falling back to English only.")
                st.exception(e)
