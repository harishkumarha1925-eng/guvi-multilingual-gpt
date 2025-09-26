# src/translation.py
from __future__ import annotations

import os
from typing import Dict, Tuple, Optional

from transformers import pipeline
from langdetect import detect, DetectorFactory

# Make langdetect deterministic
DetectorFactory.seed = 42

# ---- Model & language setup -------------------------------------------------

# Hugging Face model (override via env if you want)
NLLB_MODEL = os.getenv("NLLB_MODEL", "facebook/nllb-200-distilled-600M")

# Map langdetect ISO-639-1 codes -> NLLB language tags
LD_TO_NLLB: Dict[str, str] = {
    "en": "eng_Latn",

    # Indian languages (add more as needed)
    "hi": "hin_Deva",   # Hindi
    "ta": "tam_Taml",   # Tamil
    "te": "tel_Telu",   # Telugu
    "bn": "ben_Beng",   # Bengali
    "ml": "mal_Mlym",   # Malayalam
    "kn": "kan_Knda",   # Kannada
    "mr": "mar_Deva",   # Marathi
    "gu": "guj_Gujr",   # Gujarati
    "pa": "pan_Guru",   # Punjabi (Gurmukhi)
}

# Reverse map if you need it
NLLB_TO_NAME: Dict[str, str] = {
    "eng_Latn": "English",
    "hin_Deva": "Hindi",
    "tam_Taml": "Tamil",
    "tel_Telu": "Telugu",
    "ben_Beng": "Bengali",
    "mal_Mlym": "Malayalam",
    "kan_Knda": "Kannada",
    "mar_Deva": "Marathi",
    "guj_Gujr": "Gujarati",
    "pan_Guru": "Punjabi",
}

# Cache for translation pipelines
_TRANSLATORS: Dict[Tuple[str, str], any] = {}

# We remember the last user language so we can translate answers back
_LAST_USER_LANG_NLLB: Optional[str] = None


# ---- Helpers ----------------------------------------------------------------

def _ascii_like(text: str) -> bool:
    """Simple heuristic: True if text is mostly ASCII (likely English)."""
    if not text:
        return True
    ascii_count = sum(1 for c in text if ord(c) < 128)
    return ascii_count / max(1, len(text)) > 0.9


def detect_lang_nllb(text: str) -> str:
    """
    Detect language with langdetect and return an NLLB tag.
    Fallbacks to Tamil for non-ASCII if detection fails, else English.
    """
    try:
        if _ascii_like(text):
            return "eng_Latn"
        code = detect(text)  # e.g., 'ta', 'hi', ...
        return LD_TO_NLLB.get(code, "eng_Latn")
    except Exception:
        # Heuristic fallback: non-ASCII -> Tamil, else English
        return "tam_Taml" if not _ascii_like(text) else "eng_Latn"


def _get_translator(src: str, tgt: str):
    """
    Return a cached HF translation pipeline for src → tgt.
    """
    key = (src, tgt)
    if key not in _TRANSLATORS:
        _TRANSLATORS[key] = pipeline(
            "translation",
            model=NLLB_MODEL,
            src_lang=src,
            tgt_lang=tgt,
            max_length=512,
        )
    return _TRANSLATORS[key]


# ---- Public API --------------------------------------------------------------

def translate_text(text: str, target_lang: Optional[str] = "eng_Latn") -> str:
    """
    Translate `text` into `target_lang` using NLLB-200.

    Behavior tailored to your router:
      • When called as translate_text(user_text, "eng_Latn"):
          - auto-detect user source language
          - remember it in _LAST_USER_LANG_NLLB
          - translate to English (eng_Latn)
      • When called as translate_text(english_answer, None):
          - translate English back into the last user language
          - if last language is English or unknown, return the text as-is.

    If translation fails, returns a short error string like:
        "[Translation error: ValueError]"
    """
    global _LAST_USER_LANG_NLLB

    try:
        if not text or not text.strip():
            return ""

        # Detect source lang (NLLB tag)
        src_lang = detect_lang_nllb(text)

        # FIRST CALL (user input -> English)
        if target_lang == "eng_Latn":
            _LAST_USER_LANG_NLLB = src_lang
            if src_lang == "eng_Latn":
                # Already English
                return text
            translator = _get_translator(src_lang, "eng_Latn")
            out = translator(text)
            return out[0]["translation_text"].strip()

        # SECOND CALL (LLM English answer -> user language)
        if target_lang is None:
            tgt = _LAST_USER_LANG_NLLB or "eng_Latn"
            # If the last user language is English, no need to translate
            if tgt == "eng_Latn":
                return text
            # Here, the source should be English
            translator = _get_translator("eng_Latn", tgt)
            out = translator(text)
            return out[0]["translation_text"].strip()

        # GENERIC: explicit target given (use detection for src)
        if src_lang == target_lang:
            return text
        translator = _get_translator(src_lang, target_lang)
        out = translator(text)
        return out[0]["translation_text"].strip()

    except Exception as e:
        return f"[Translation error: {type(e).__name__}]"

