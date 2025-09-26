# src/llm_backend.py (excerpt)

def generate_answer(prompt: str, domain_role: str = "general") -> str:
    """
    Generate an answer from the LLM in English.

    Parameters
    ----------
    prompt : str
        The user input, already translated to English.
    domain_role : str
        The domain mode (general, mentor, faq, recommender, etc.)

    Returns
    -------
    str
        Always returns a string (never None), with a fallback error message if LLM fails.
    """
    try:
        # Call your internal local generator (already defined elsewhere in this file)
        response = _local_generate(prompt)

        # Force everything into string to avoid TypeError
        return str(response).strip() if response else "[LLM error: empty response]"

    except Exception as e:
        # Always return a string, even on error
        return f"[LLM error: {type(e).__name__}]"


