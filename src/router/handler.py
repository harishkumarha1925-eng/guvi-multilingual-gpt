# src/router/handler.py
from src.translation import translate_text, detect_language
from src.llm_backend import generate_answer

def handle_turn(user_text: str, domain_role: str = "general") -> str:
    try:
        # Detect input language
        user_lang = detect_language(user_text)

        # Translate user input to English for LLM
        eng = translate_text(user_text, target_lang="eng_Latn")
        if not eng:
            return "⚠️ Could not translate your input."

        # Generate answer in English
        reply_en = generate_answer(eng, domain_role=domain_role)
        if not reply_en:
            return "⚠️ The language model did not return a response."

        # Translate answer back into user's language
        if user_lang != "en":
            reply_back = translate_text(reply_en, target_lang=user_lang)
            return reply_back or reply_en
        else:
            return reply_en
    except Exception as e:
        return f"⚠️ Internal error: {type(e).__name__}"

