# src/translation.py
from __future__ import annotations

import re
from typing import Optional

# ---- Robust import of langdetect (won't crash the app if missing) ----------
try:
    from langdetect import detect, DetectorFactory

    DetectorFactory.seed = 0  # make detection deterministic
    _LANGDETECT = True
except Exception:
    detect = None  # type: ignore
    _LANGDETECT = False


# ---- Language codes (NLLB style codes we display/use elsewhere) ------------
ENG = "eng_Latn"
TAM = "tam_Taml"
HIN = "hin_Deva"
TEL = "tel_Telu"
MAL = "mal_Mlym"
BEN = "ben_Beng"

# Map langdetect's 2-letter tags to our codes
DETECT_TO_CODE = {
    "en": ENG,
    "ta": TAM,
    "hi": HIN,
    "te": TEL,
    "ml": MAL,
    "bn": BEN,
}

# Keep track of the last user's language so we can translate the answer back.
_last_user_lang: str = ENG


# ---- Utilities --------------------------------------------------------------
_ws_re = re.compile(r"\s+")
_punct_re = re.compile(r"[^\w\s]+")


def _normalize(s: str) -> str:
    s = s.strip().lower()
    s = _punct_re.sub("", s)
    s = _ws_re.sub(" ", s)
    return s


def _safe_detect(text: str) -> Optional[str]:
    """Return 2-letter ISO code if possible."""
    if _LANGDETECT and detect:
        try:
            code = detect(text)
            if isinstance(code, str):
                return code
        except Exception:
            return None
    return None


def _code_from_detect(text: str) -> str:
    """Return our internal code (ENG/TAM/…) from text; default ENG."""
    iso2 = _safe_detect(text)
    if iso2 and iso2 in DETECT_TO_CODE:
        return DETECT_TO_CODE[iso2]
    return ENG


# ---- Tiny phrase dictionaries for demo translation -------------------------
# NOTE: These are intentionally small. Unknown phrases simply echo through.

# English -> Indic
EN_TO_TA = {
    "hello": "வணக்கம்",
    "thank you": "நன்றி",
    "what is the capital of japan": "ஜப்பானின் தலைநகரம் என்ன",
    "who are you": "நீங்கள் யார்",
}
EN_TO_HI = {
    "hello": "नमस्ते",
    "thank you": "धन्यवाद",
    "what is the capital of japan": "जापान की राजधानी क्या है",
    "who are you": "आप कौन हैं",
}
EN_TO_TE = {
    "hello": "నమస్తే",
    "thank you": "ధన్యవాదాలు",
    "what is the capital of japan": "జపాన్ రాజధాని ఏమిటి",
    "who are you": "మీరు ఎవరు",
}
EN_TO_ML = {
    "hello": "നമസ്കാരം",
    "thank you": "നന്ദി",
    "what is the capital of japan": "ജപ്പാന്റെ തലസ്ഥാനം ഏത്",
    "who are you": "താങ്കള്‍ ആര്",
}
EN_TO_BN = {
    "hello": "নমস্কার",
    "thank you": "ধন্যবাদ",
    "what is the capital of japan": "জাপানের রাজধানী কী",
    "who are you": "আপনি কে",
}

# Indic -> English (rough)
TA_TO_EN = {
    "வணக்கம்": "hello",
    "நன்றி": "thank you",
    "ஜப்பானின் தலைநகரம் என்ன": "what is the capital of japan",
    "நீங்கள் யார்": "who are you",
}
HI_TO_EN = {
    "नमस्ते": "hello",
    "धन्यवाद": "thank you",
    "जापान की राजधानी क्या है": "what is the capital of japan",
    "आप कौन हैं": "who are you",
}
TE_TO_EN = {
    "నమస్తే": "hello",
    "ధన్యవాదాలు": "thank you",
    "జపాన్ రాజధాని ఏమిటి": "what is the capital of japan",
    "మీరు ఎవరు": "who are you",
}
ML_TO_EN = {
    "നമസ്കാരം": "hello",
    "നന്ദി": "thank you",
    "ജപ്പാന്റെ തലസ്ഥാനം ഏത്": "what is the capital of japan",
    "താങ്കള്‍ ആര്": "who are you",
}
BN_TO_EN = {
    "নমস্কার": "hello",
    "ধন্যবাদ": "thank you",
    "জাপানের রাজধানী কী": "what is the capital of japan",
    "আপনি কে": "who are you",
}

EN_TO_MAP = {
    TAM: EN_TO_TA,
    HIN: EN_TO_HI,
    TEL: EN_TO_TE,
    MAL: EN_TO_ML,
    BEN: EN_TO_BN,
}

TO_EN_MAP = {
    TAM: TA_TO_EN,
    HIN: HI_TO_EN,
    TEL: TE_TO_EN,
    MAL: ML_TO_EN,
    BEN: BN_TO_EN,
}


def _en_to_lang(text: str, target_lang: str) -> str:
    """Best-effort phrase translation from English to target language."""
    norm = _normalize(text)
    table = EN_TO_MAP.get(target_lang)
    if not table:
        return text  # unknown lang: echo
    return table.get(norm, text)


def _lang_to_en(text: str, src_lang: str) -> str:
    """Best-effort phrase translation from source language to English."""
    table = TO_EN_MAP.get(src_lang)
    if not table:
        return text  # unknown lang: echo
    # try exact first, then normalized key search
    if text in table:
        return table[text]
    norm = _normalize(text)
    for k, v in table.items():
        if _normalize(k) == norm:
            return v
    return text


# ---- Public API -------------------------------------------------------------
def translate_text(text: str, target_lang: Optional[str] = None) -> str:
    """
    Translate text between English and supported Indic languages.

    - If target_lang == "eng_Latn": translate input to English (best effort).
    - If target_lang is None: translate English answer back to the user's
      last detected language (tracked per process).
    - If target_lang is one of (tam/hin/tel/mal/ben codes): translate from
      English into that language.
    - Unknown phrases or unsupported languages are returned unchanged.
    """
    global _last_user_lang

    if not text or not text.strip():
        return text

    # 1) To English
    if target_lang == ENG:
        src_code = _code_from_detect(text)
        # Remember user's language for the return trip
        _last_user_lang = src_code
        if src_code == ENG:
            return text
        return _lang_to_en(text, src_code)

    # 2) Translate back to the last user's language (if any)
    if target_lang is None:
        tgt = _last_user_lang or ENG
        if tgt == ENG:
            return text
        return _en_to_lang(text, tgt)

    # 3) Explicit English -> target language
    return _en_to_lang(text, target_lang)


def get_last_user_lang() -> str:
    """Expose last detected user lang code (useful for debugging/UI)."""
    return _last_user_lang

