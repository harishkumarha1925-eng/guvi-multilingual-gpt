# app.py
"""
GUVI Multilingual GPT Chatbot (Streamlit)
- Translates user input to English (internal pivot)
- Retrieves top-k GUVI passages (RAG) using sentence-transformers
- Builds a prompt with context and queries a remote model (Hugging Face or OpenAI)
- Translates model output back to user's target language

Requirements:
- utils.py containing translate_text, hf_inference, openai_generate,
  retrieve, build_prompt_with_context, ensure_index (as provided earlier)
- guvi_data.txt in the repo root (or pre-built index in index/)
"""

import streamlit as st
import os
from utils import (
    translate_text,
    hf_inference,
    openai_generate,
    retrieve,
    build_prompt_with_context,
    ensure_index,
)

st.set_page_config(page_title="GUVI Multilingual GPT Chatbot", page_icon="🤖", layout="wide")

# --- Sidebar: configuration, secrets & index building ---
st.sidebar.header("Configuration & Secrets")
st.sidebar.info(
    "Provide either a Hugging Face model+token or an OpenAI API key. "
    "Prefer storing keys in Streamlit Secrets (recommended)."
)

# Inputs for tokens/models (can be left blank to use st.secrets)
hf_model = st.sidebar.text_input("Hugging Face model (e.g. username/guvi-model)", value="")
hf_token = st.sidebar.text_input("Hugging Face token (leave blank to use secrets)", type="password")
openai_key = st.sidebar.text_input("OpenAI API key (optional)", type="password")

# read secrets if inputs empty
if not hf_token and "HF_TOKEN" in st.secrets:
    hf_token = st.secrets["HF_TOKEN"]
if not openai_key and "OPENAI_API_KEY" in st.secrets:
    openai_key = st.secrets["OPENAI_API_KEY"]

# Index options
st.sidebar.markdown("---")
st.sidebar.subheader("Index / Embeddings")
index_build_now = st.sidebar.button("(Re)build embeddings from guvi_data.txt")
show_index_info = st.sidebar.checkbox("Show index status", value=False)

# If user requests, ensure index is built
if index_build_now:
    with st.spinner("Building or refreshing index (this may take a while)..."):
        try:
            corpus, embeddings, _ = ensure_index()
            st.sidebar.success(f"Index ready: {len(corpus)} documents.")
        except Exception as e:
            st.sidebar.error(f"Index build failed: {e}")

# optionally show index status
if show_index_info:
    try:
        corpus_preview, embeddings, _ = ensure_index()
        st.sidebar.write(f"Documents in index: {len(corpus_preview)}")
        if len(corpus_preview) > 0:
            st.sidebar.markdown("Example doc (first):")
            st.sidebar.write(corpus_preview[0][:400] + ("..." if len(corpus_preview[0]) > 400 else ""))
    except Exception as e:
        st.sidebar.warning(f"Index not ready: {e}")

# --- App UI ---
st.title("GUVI Multilingual GPT Chatbot")
st.markdown(
    "Ask GUVI-specific questions in any supported language. The app translates your question, "
    "retrieves relevant GUVI passages, sends a context-aware prompt to the model, and returns "
    "a translated reply."
)

col_main, col_side = st.columns([3, 1])

with col_main:
    user_input = st.text_area("Your message", height=160, placeholder="Type your question here (any language).")
    if not user_input:
        st.caption("Tip: ask something about GUVI (courses, enrollment, contact, etc.) to see domain-aware answers.")
    send_button = st.button("Send")

with col_side:
    src_lang = st.selectbox(
        "Input language",
        ["auto", "en", "hi", "ta", "te", "ml", "bn", "kn", "mr", "gu", "pa", "ur", "fr", "es", "de"],
        index=0,
    )
    tgt_lang = st.selectbox(
        "Output language",
        ["en", "hi", "ta", "te", "ml", "bn", "kn", "mr", "gu", "pa", "ur", "fr", "es", "de"],
        index=0,
    )
    max_tokens = st.slider("Max model tokens", min_value=50, max_value=1024, value=256, step=50)
    top_k = st.slider("Top-k retrieved passages", min_value=1, max_value=5, value=3)

# --- Main logic ---
if send_button:
    if not user_input.strip():
        st.warning("Please type a message before sending.")
        st.stop()

    # 1) Translate input -> English (internal)
    try:
        with st.spinner("Translating input..."):
            model_input = translate_text(user_input, source_lang=src_lang, target_lang="en")
    except Exception as e:
        st.error(f"Translation failed: {e}")
        st.stop()

    # 2) Retrieve domain context
    try:
        with st.spinner("Retrieving domain context..."):
            retrieved = retrieve(model_input, top_k=top_k)
            passages = [p for p, score in retrieved]
    except Exception as e:
        st.error(f"Retrieval failed: {e}")
        passages = []

    # 3) Build prompt with context and question
    prompt_for_model = build_prompt_with_context(model_input, passages)

    # 4) Query model (Hugging Face preferred, fallback to OpenAI)
    response_text = None
    # Preferred: Hugging Face Inference API
    if hf_model and hf_token:
        try:
            with st.spinner("Querying Hugging Face model..."):
                response_text = hf_inference(hf_model, prompt_for_model, hf_token, max_length=max_tokens)
        except Exception as e:
            st.error(f"Hugging Face inference failed: {e}")

    # Fallback: OpenAI
    if response_text is None and openai_key:
        try:
            with st.spinner("Querying OpenAI..."):
                response_text = openai_generate(prompt_for_model, openai_key, max_tokens=max_tokens)
        except Exception as e:
            st.error(f"OpenAI call failed: {e}")

    # If no remote model configured, produce a helpful fallback reply that includes context
    if response_text is None:
        response_text = (
            f"(No model configured) Would reply to: {model_input}\n\n"
            "Context included:\n"
            + ("\n\n".join(passages) if passages else "(no context retrieved)")
        )

    # 5) Translate model output -> user's target language
    try:
        with st.spinner("Translating output..."):
            final_reply = translate_text(response_text, source_lang="en", target_lang=tgt_lang)
    except Exception as e:
        st.error(f"Output translation failed: {e}")
        final_reply = response_text

    # 6) Show results
    st.subheader("Reply")
    st.markdown(final_reply)

    # Debug / transparency panel
    with st.expander("Debug data (show raw inputs & retrievals)"):
        st.write("**Original input:**", user_input)
        st.write("**Model input (EN):**", model_input)
        st.write("**Prompt sent to model (truncated):**")
        st.code(prompt_for_model[:3000] + ("..." if len(prompt_for_model) > 3000 else ""), language="text")
        st.write("**Model raw response:**", response_text)
        if passages:
            st.write("**Retrieved passages (top-k):**")
            for i, (p, score) in enumerate(retrieved, 1):
                st.markdown(f"**{i}. score={score:.4f}**")
                st.write(p)
        else:
            st.write("No passages retrieved.")

# End of app
