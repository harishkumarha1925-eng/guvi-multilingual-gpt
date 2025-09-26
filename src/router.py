# src/router.py
from .language_detection import detect_lang_code
from .translation import maybe_translate_to_english, translate_back
from .llm_backend import generate_answer
from .heuristics import maybe_answer_locally  # <-- add this import

def handle_turn(user_text: str, domain_role: str = "general"):
    iso, nllb_src = detect_lang_code(user_text)
    english_text, translated_in = maybe_translate_to_english(user_text, nllb_src)

    # NEW: try to answer without LLM for simple utilities
    local = maybe_answer_locally(english_text)
    if local is not None:
        final_answer = translate_back(local, nllb_src)
        return {
            "user_iso": iso,
            "user_nllb": nllb_src,
            "was_translated_in": translated_in,
            "english_query": english_text,
            "english_answer": local,
            "final_answer": final_answer,
            "answered_via": "local_heuristic",
        }

    english_answer = generate_answer(english_text, domain_role=domain_role)
    final_answer = translate_back(english_answer, nllb_src)
    return {
        "user_iso": iso,
        "user_nllb": nllb_src,
        "was_translated_in": translated_in,
        "english_query": english_text,
        "english_answer": english_answer,
        "final_answer": final_answer,
        "answered_via": "llm",
    }
