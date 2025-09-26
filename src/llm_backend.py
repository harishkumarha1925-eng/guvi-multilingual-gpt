# src/llm_backend.py
"""
LLM backends for the GUVI Multilingual GPT Chatbot.

- local_small: uses FLAN-T5 locally (CPU-friendly). Downloads PUBLIC models anonymously
               (no Authorization header) to avoid 401 errors from stale/invalid tokens.
- hf_inference: uses Hugging Face Inference API for hosted instruct models.

Environment variables are read from src.config.settings.
"""

from __future__ import annotations

import os
import json
import time
from functools import lru_cache
from typing import Dict

import requests
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from src.config import settings

# Guard: never allow hf_inference without a valid token
if settings.LLM_MODE == "hf_inference" and not settings.HF_API_TOKEN:
    # fallback to local automatically
    settings.LLM_MODE = "local_small"


# ------------- Shared prompt scaffolding -------------
SYSTEM_PREFIX = (
    "You are a helpful, concise multilingual assistant for GUVI. "
    "Answer accurately. If asked for GUVI courses/recommendations, propose relevant options."
)


def _role_prefix(domain_role: str) -> str:
    return {
        "general": "",
        "faq": "You answer FAQs about GUVI courses, pricing, enrollment, and platform usage. Be brief.",
        "mentor": "You act as a career mentor. Ask 1-2 clarifying questions only if essential, then give a plan.",
        "recommender": "You recommend GUVI courses/resources tailored to interests and skill levels.",
    }.get(domain_role, "")


def _build_prompt(user_english_text: str, domain_role: str) -> str:
    return f"{SYSTEM_PREFIX}\n{_role_prefix(domain_role)}\nUser: {user_english_text}\nAssistant:"


# ------------- Utilities -------------
def _strip_hf_tokens_from_env() -> None:
    for var in ("HUGGINGFACE_HUB_TOKEN", "HF_TOKEN"):
        os.environ.pop(var, None)

def _anon_kwargs_for_transformers() -> dict:
    # Use ONLY with from_pretrained; DO NOT pass to pipeline(...)
    return {"token": None, "use_auth_token": None, "trust_remote_code": False}

@lru_cache(maxsize=1)
def _get_local_small_pipeline():
    _strip_hf_tokens_from_env()

    model_candidates = [
        "google/flan-t5-large",
        "google/flan-t5-base",
        "google/flan-t5-small",
    ]

    last_err = None
    for model_id in model_candidates:
        try:
            # Load tokenizer + model anonymously
            kw = _anon_kwargs_for_transformers()
            tok = AutoTokenizer.from_pretrained(model_id, **kw)
            mdl = AutoModelForSeq2SeqLM.from_pretrained(model_id, **kw)

            # Build pipeline WITHOUT extra kwargs (so they don't leak to forward())
            gen = pipeline("text2text-generation", model=mdl, tokenizer=tok)

            # Light sanity inference to force weights init
            _ = gen("Hello", max_new_tokens=4)
            return gen
        except Exception as e:
            last_err = e

    raise RuntimeError(
        "Failed to load a local FLAN-T5 pipeline. Last error:\n"
        f"{type(last_err).__name__}: {last_err}\n\n"
        "Troubleshooting tips:\n"
        " • Ensure you removed/emptied HF tokens in .env and from your environment.\n"
        " • Check your internet connection (first download caches the model).\n"
        " • If a corporate proxy is used, set HTTP(S)_PROXY env vars.\n"
        " • As a last resort, set LLM_MODE=hf_inference with a valid HF token."
    )

def _local_generate(prompt: str, max_new_tokens: int = 256) -> str:
    gen = _get_local_small_pipeline()
    out = gen(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
    )
    return (out[0]["generated_text"] if out else "").strip()


# ------------- HF Inference API backend -------------
def _hf_inference_generate(prompt: str, max_new_tokens: int = 256) -> str:
    """
    Calls the Hugging Face Inference API.
    Requires:
      - settings.HF_API_TOKEN (valid)
      - settings.HF_TEXT_GENERATION_MODEL (e.g., meta-llama/Meta-Llama-3.1-8B-Instruct)
    """
    if not settings.HF_API_TOKEN:
        raise RuntimeError("LLM_MODE=hf_inference but HF_API_TOKEN is not set in your .env or secrets.")
    model_id = settings.HF_TEXT_GENERATION_MODEL
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {settings.HF_API_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
            "return_full_text": False,
        }
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=120)
        # Raise HTTP errors with context
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data and "generated_text" in data[0]:
            return data[0]["generated_text"].strip()
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"].strip()
        # Some hosted models return a list of dicts with 'generated_text'
        return json.dumps(data)[:2000]
    except requests.HTTPError as e:
        # Common cold-start or loading messages from HF endpoint
        text = ""
        try:
            text = r.text  # type: ignore
        except Exception:
            pass
        raise RuntimeError(
            f"Hugging Face Inference API error ({r.status_code if 'r' in locals() else 'HTTP'}): {e}\n"
            f"Response: {text}"
        ) from e
    except requests.RequestException as e:
        raise RuntimeError(f"Network error calling HF Inference API: {e}") from e


# ------------- Public API -------------
def generate_answer(user_english_text: str, domain_role: str = "general") -> str:
    """
    Parameters
    ----------
    user_english_text : str
        The (already translated) user prompt in English.
    domain_role : str
        One of: general | faq | mentor | recommender

    Returns
    -------
    str : model response (English)
    """
    prompt = _build_prompt(user_english_text, domain_role)

    # If user configured API mode, use it; else local small
    mode = (settings.LLM_MODE or "local_small").strip().lower()
    try:
        if mode == "hf_inference":
            return _hf_inference_generate(prompt)
        # default / fallback
        return _local_generate(prompt)
    except Exception as e:
        # Provide a concise, user-friendly error; log the rest in Streamlit UI if needed.
        raise RuntimeError(
            "The language model failed to generate a response. "
            f"Mode: {mode}. Details: {type(e).__name__}: {e}"
        ) from e


