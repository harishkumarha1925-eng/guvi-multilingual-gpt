# src/llm_backend.py
from typing import Optional
from transformers import pipeline

# cache for local pipeline
_llm: Optional[pipeline] = None


def _get_llm():
    """Initialize the small local model pipeline once (CPU-friendly)."""
    global _llm
    if _llm is None:
        _llm = pipeline(
            "text2text-generation",
            model="google/flan-t5-small",   # small & fast
            tokenizer="google/flan-t5-small",
            device_map="cpu",
        )
    return _llm


def _local_generate(prompt: str, domain_role: str = "general") -> str:
    """
    Generate an answer using FLAN-T5-small locally.
    Clean prompt template: avoid 'You are...' instructions that get echoed.
    """
    llm = _get_llm()

    # Role-specific style guides
    styles = {
        "general":    "Answer the question in one short sentence.",
        "technical":  "Answer precisely and briefly.",
        "educational":"Explain in one short sentence.",
        "friendly":   "Answer briefly and warmly.",
    }
    style = styles.get(domain_role, styles["general"])

    instruction = (
        f"{style}\n\n"
        f"Question: {prompt}\n"
        f"Answer:"
    )

    out = llm(
        instruction,
        max_new_tokens=64,
        do_sample=False,
        early_stopping=True,
        num_return_sequences=1,
    )[0]["generated_text"]

    return str(out).strip()


def generate_answer(prompt: str, domain_role: str = "general") -> str:
    """
    Public entry point used by router. Always returns a string.
    """
    try:
        response = _local_generate(str(prompt), domain_role=str(domain_role))
        return str(response)
    except Exception as e:
        return f"[LLM error: {type(e).__name__}]"






