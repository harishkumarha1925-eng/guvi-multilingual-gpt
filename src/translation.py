# src/translation.py
"""
Translation utilities for the GUVI Multilingual GPT Chatbot.

- Loads NLLB-200 anonymously (no HF token sent) to avoid 401 errors.
- Caches tokenizer/model for performance.
- Provides helpers to translate to English and back to the user's language.

Env/config is read from src.config.settings.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Tuple

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .config import settings


# -------------------- Internal helpers --------------------
def _strip_hf_tokens_from_env() -> None:
    """Ensure public model downloads NEVER send an Authorization header."""
    for var in ("HUGGINGFACE_HUB_TOKEN", "HF_TOKEN"):
        os.environ.pop(var, None)


def _anon_kwargs_for_transformers() -> dict:
    """
    kwargs that force anonymous download across transformers versions.
    - Newer: token=None
    - Older: use_auth_token=None
    """
    return {"token": None, "use_auth_token": None, "trust_remote_code": False}


def _forced_bos_id(tokenizer, lang_code: str) -> int:
    """
    Return the correct forced BOS token id for a given NLLB language code.
    Works across different tokenizer variants.
    """
    # Preferred mapping on many NLLB tokenizers
    if hasattr(tokenizer, "lang_code_to_id") and lang_code in tokenizer.lang_code_to_id:
        return tokenizer.lang_code_to_id[lang_code]

    # Some tokenizers expose convert_tokens_to_ids for lang codes
    try:
        return tokenizer.convert_tokens_to_ids(lang_code)  # type: ignore[attr-defined]
    except Exception:
        pass

    # Fallback (English Latin script)
    # Note: this is safe but will make generation English if the mapping isn't found.
    return tokenizer.lang_code_to_id.get("eng_Latn", tokenizer.eos_token_id)


# -------------------- Model / Tokenizer singletons --------------------
@lru_cache(maxsize=1)
def _get_nllb():
    """
    Lazily load and cache the NLLB tokenizer and model, with anonymous downloads.
    """
    _strip_hf_tokens_from_env()
    kwargs = _anon_kwargs_for_transformers()

    tok = AutoTokenizer.from_pretrained(settings.NLLB_MODEL, **kwargs)
    model = AutoModelForSeq2SeqLM.from_pretrained(settings.NLLB_MODEL, **kwargs)
    return tok, model


# -------------------- Public translation API --------------------
def translate(text: str, src_lang: str, tgt_lang: str) -> str:
    """
    Translate `text` from src_lang (NLLB code) to tgt_lang (NLLB code).
    Note: The underlying generation only needs `forced_bos_token_id` for the target.
    """
    tok, model = _get_nllb()

    inputs = tok(text, return_tensors="pt", truncation=True)
    bos_id = _forced_bos_id(tok, tgt_lang)

    generated = model.generate(
        **inputs,
        forced_bos_token_id=bos_id,
        max_new_tokens=512,
        num_beams=3,
        early_stopping=True,
    )
    return tok.decode(generated[0], skip_special_tokens=True)


def maybe_translate_to_english(text: str, src_nllb: str) -> Tuple[str, bool]:
    """
    If the source is not English (eng_Latn), translate to English and flag True.
    Otherwise return the original text and flag False.
    """
    if src_nllb == "eng_Latn":
        return text, False
    return translate(text, src_nllb, "eng_Latn"), True


def translate_back(text: str, tgt_nllb: str) -> str:
    """
    Translate English `text` back to the user's target language (NLLB code).
    If the target is English, return the text unchanged.
    """
    if tgt_nllb == "eng_Latn":
        return text
    return translate(text, "eng_Latn", tgt_nllb)
