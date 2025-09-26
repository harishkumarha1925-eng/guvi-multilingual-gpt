# src/router/handler.py
from src.translation import translate_text
from src.llm_backend import generate_answer

def handle_turn(user_text: str, domain_role: str = "general") -> str:
    try:
        # auto-detect user language and translate to English
        eng = translate_text(user_text, target_lang="eng_Latn")
        if not eng:
            return "⚠️ Could not translate your input."

        reply_en = generate_answer(eng, domain_role=domain_role)
        if not reply_en:
            return "⚠️ The language model did not return a response."

        # translate back to the user's language (None = auto)
        reply_back = translate_text(reply_en, target_lang=None)
        return reply_back or reply_en
    except Exception as e:
        return f"⚠️ Internal error: {type(e).__name__}"

