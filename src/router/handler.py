# src/router/handler.py
from src.translation import translate_text, ENG
from src.llm_backend import generate_answer

def handle_turn(user_text: str, domain_role: str = "general") -> str:
    try:
        # Translate the user's input to English (this also updates the last-user-lang internally)
        english_text = translate_text(user_text, target_lang=ENG)
        if not english_text:
            return "⚠️ Could not translate your input."

        # Get an English answer from the LLM
        english_answer = generate_answer(english_text, domain_role=domain_role)
        if not english_answer:
            return "⚠️ The language model did not return a response."

        # Translate the answer back to the user's last detected language
        final = translate_text(english_answer, target_lang=None)
        return final or english_answer

    except Exception as e:
        return f"⚠️ Internal error: {type(e).__name__}"


