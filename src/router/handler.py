from src.translation import translate_text
from src.llm_backend import generate_answer

def handle_turn(user_text: str, domain_role: str = "general") -> str:
    try:
        # Translate input to English
        english_text = translate_text(user_text, target_lang="eng_Latn")
        if not english_text:
            return "⚠️ Could not translate your input."

        english_text = str(english_text)

        # Generate answer in English
        english_answer = generate_answer(english_text, domain_role=domain_role)
        if not english_answer:
            return "⚠️ The language model did not return a response."

        english_answer = str(english_answer)

        # Translate back to user language
        translated_back = translate_text(english_answer, target_lang=None)
        return str(translated_back) if translated_back else english_answer

    except Exception as e:
        return f"⚠️ Internal error: {type(e).__name__}"
