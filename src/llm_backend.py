def generate_answer(prompt: str, domain_role: str = "general") -> str:
    """
    Generate an answer from the LLM in English.
    Always returns a string.
    """
    try:
        response = _local_generate(prompt)
        return str(response).strip() if response else "[LLM error: empty response]"
    except Exception as e:
        return f"[LLM error: {type(e).__name__}]"



