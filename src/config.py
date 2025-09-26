import os

# Streamlit Cloud injects secrets via st.secrets, but we keep this file
# framework-agnostic so it also works locally.
def _get(key: str, default: str = "") -> str:
    # 1) Streamlit secrets (if present)
    try:
        import streamlit as st  # will exist on Streamlit Cloud
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    # 2) Environment
    return os.getenv(key, default)

# ---- raw values from env/secrets
LLM_MODE = _get("LLM_MODE", "local_small").strip()
NLLB_MODEL = _get("NLLB_MODEL", "facebook/nllb-200-distilled-600M").strip()
NLLB_SRC_LANG = _get("NLLB_SRC_LANG", "auto").strip()
NLLB_TGT_LANG = _get("NLLB_TGT_LANG", "eng_Latn").strip()
APP_TITLE = _get("APP_TITLE", "GUVI Multilingual GPT Chatbot").strip()

# HF token (if someone added one in secrets by mistake)
HF_API_TOKEN = _get("HF_API_TOKEN", "").strip()

# ---- normalization & safety switches ----------------------------------------
# Normalize mode
LLM_MODE = LLM_MODE.lower().strip()

# If no HF token, or token looks redacted/invalid, force local_small
if LLM_MODE == "hf_inference":
    if not HF_API_TOKEN or HF_API_TOKEN.lower() in {"", "none", "null", "redacted"}:
        LLM_MODE = "local_small"

# Final, exported settings object-like shim (to keep existing imports working)
class settings:  # noqa: N801 (match existing code style)
    LLM_MODE = LLM_MODE
    NLLB_MODEL = NLLB_MODEL
    NLLB_SRC_LANG = NLLB_SRC_LANG
    NLLB_TGT_LANG = NLLB_TGT_LANG
    APP_TITLE = APP_TITLE
    HF_API_TOKEN = HF_API_TOKEN
