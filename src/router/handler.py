# src/router/handler.py
from ..translation import translate_text, detect_language  # ← relative import
from ..llm_backend import generate_answer

def handle_turn(user_text: str, domain_role: str = "general") -> str:
    """
    1) Detect input language
    2) Translate to English
    3) Generate answer in English
    4) Translate back to user's language
    """
    try:
        # Detect source language (BCP-47 / ISO-ish)
        src_lang = detect_language(user_text) or "eng_Latn"

        # To English for the LLM
        english_text = translate_text(user_text, target_lang="eng_Latn")
        if not english_text:
            return "⚠️ Could not translate your input."

        # LLM generation in English
        english_answer = generate_answer(english_text, domain_role=domain_role)
        if not english_answer:
            return "⚠️ The language model did not return a response."

        # Back to user language
        translated_back = translate_text(english_answer, target_lang=src_lang)
        return translated_back or english_answer

    except Exception as e:
        return f"⚠️ Internal error: {type(e).__name__}"

