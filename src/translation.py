# src/translation.py
from typing import Any, Optional

# if you had a real translator, import it here
# from ._your_translator_impl import translator

def _as_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        # common HF keys
        for k in ("translation_text", "generated_text", "text", "response", "answer"):
            if k in x and isinstance(x[k], str):
                return x[k]
        # last resort
        try:
            import json
            return json.dumps(x)
        except Exception:
            return str(x)
    if isinstance(x, (list, tuple)) and x:
        return _as_text(x[0])
    return str(x)

def translate_text(text: str, target_lang: Optional[str] = None) -> str:
    """
    Return translated text as a string. If translation fails, return the input text.
    target_lang can be like 'eng_Latn' or None to auto/identity.
    """
    try:
        # ---- replace this block with your actual translator call ----
        # demo: just echo text (identity translation) so the app works
        out = text
        # e.g. out = translator(text, target_lang=target_lang)
        # -------------------------------------------------------------
        return _as_text(out).strip()
    except Exception:
        # Fail safe: return original text
        return text or ""

