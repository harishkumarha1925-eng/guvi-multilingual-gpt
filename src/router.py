# src/router.py
from src.translation import translate_text
from src.llm_backend import generate_answer

def handle_turn(user_text: str, domain_role: str = "general") -> str:
    try:
        # 1) to English
        english_text = translate_text(user_text, target_lang="eng_Latn") or ""
        if not english_text.strip():
            return "⚠️ Could not translate your input."

        # 2) get answer
        english_answer = generate_answer(english_text, domain_role=str(domain_role or "general"))
        if not english_answer.strip():
            return "⚠️ The language model did not return a response."

        # 3) back to user language (identity if your translator is a stub)
        final_text = translate_text(english_answer, target_lang=None) or english_answer
        return final_text.strip() or english_answer
    except Exception as e:
        return f"⚠️ Internal error: {type(e).__name__}"


