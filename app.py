# app.py
import os
from types import SimpleNamespace
import streamlit as st

# ── Safe imports with sensible fallbacks ──────────────────────────────────────
# settings
try:
    from src.config import settings  # your own config object
except Exception:
    # Fallback defaults so the app can still boot
    settings = SimpleNamespace(
        APP_TITLE="Multilingual GPT",
        APP_DESCRIPTION="Ask anything. I’ll detect the language and reply helpfully.",
        LLM_MODE=os.getenv("LLM_MODE", "offline"),
        DOMAINS=["general", "mentor", "recommender", "faq"],
        DEFAULT_DOMAIN="general",
    )

# router / handler
try:
    # Prefer an explicit handler module if you created one
    try:
        from src.router.handler import handle_turn
    except Exception:
        from src.router import handle_turn  # type: ignore
except Exception:
    # Last-ditch fallback: echo handler so the UI still works
    def handle_turn(user_text: str, *, domain: str = "general") -> str:
        return f"[stub] Domain={domain}\nYou said: {user_text}"

# toasts
try:
    from src.utils import toast_ok, toast_warn  # optional niceties
except Exception:
    def toast_ok(msg: str) -> None:
        st.toast(msg, icon="✅")

    def toast_warn(msg: str) -> None:
        st.toast(msg, icon="⚠️")


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=getattr(settings, "APP_TITLE", "Multilingual GPT"),
    page_icon="🌍",
    layout="centered",
)

st.title(getattr(settings, "APP_TITLE", "Multilingual GPT"))
if desc := getattr(settings, "APP_DESCRIPTION", ""):
    st.caption(desc)

col1, col2 = st.columns(2)
with col1:
    st.markdown(
        f"**LLM Mode**: `{getattr(settings, 'LLM_MODE', 'unknown')}`"
    )
with col2:
    st.markdown(
        "**Status**: ready"
    )

st.divider()

# ── Domain selector & helpful examples ────────────────────────────────────────
domains = list(getattr(settings, "DOMAINS", ["general", "mentor", "recommender", "faq"]))
default_domain = getattr(settings, "DEFAULT_DOMAIN", "general")
if default_domain not in domains:
    default_domain = domains[0]

with st.sidebar:
    st.header("Settings")
    domain = st.selectbox("Domain mode", options=domains, index=domains.index(default_domain))
    st.caption("Choose the assistant's behavior for your query.")

    with st.expander("Examples", expanded=False):
        examples = {
            "general": [
                "Translate this to Spanish: I forgot my umbrella.",
                "Summarize: https://en.wikipedia.org/wiki/Streamlit",
                "Explain TF-IDF like I’m 12."
            ],
            "mentor": [
                "I’m stuck on binary search. What’s the bug in this code? (paste code)",
                "Plan a 2-week path to learn Pandas with daily tasks.",
                "Mock interview: arrays & strings, medium difficulty."
            ],
            "recommender": [
                "Suggest 3 laptops under $900 for data science beginners.",
                "Pick a Python book for absolute newbies and explain why.",
                "Top 5 open datasets for NLP beginners."
            ],
            "faq": [
                "What is this app and how does it work?",
                "What models or libraries does it use?",
                "Is my data stored?"
            ],
        }
        for ex in examples.get(domain, []):
            st.code(ex, language="text")

# ── Simple chat state ─────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []  # list[dict(role,str; content,str)]

# Chat UI
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_text = st.chat_input("Type your message in any language…")

if user_text:
    st.session_state.history.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    # Use Streamlit's official context manager (your previous error was here)
    with st.status("Thinking & translating…", expanded=False) as s:
        s.update(label="Working…", state="running")
        try:
            reply = handle_turn(user_text, domain=domain)
            s.update(label="Done", state="complete")
            toast_ok("Response ready")
        except Exception as e:
            # Show a concise error and keep the app alive
            toast_warn("Something went wrong while generating the answer.")
            reply = f"Sorry—there was an internal error: `{type(e).__name__}`. Please try again."
    st.session_state.history.append({"role": "assistant", "content": reply})

    with st.chat_message("assistant"):
        st.markdown(reply)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
with st.expander("About this app"):
    st.markdown(
        "- Multilingual input/output.\n"
        "- Domain modes adjust tone and structure of answers.\n"
        "- This UI uses only Streamlit built-ins for stability."
    )

