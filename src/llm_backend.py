# src/llm_backend.py
from typing import Optional
from transformers import pipeline

# Build the pipeline once (CPU)
_llm: Optional[pipeline] = None

def _get_llm():
    global _llm
    if _llm is None:
        # small, CPU-friendly instruction model
        _llm = pipeline(
            "text2text-generation",
            model="google/flan-t5-small",
            tokenizer="google/flan-t5-small",
            device_map="cpu",
        )
    return _llm

def _local_generate(prompt: str, domain_role: str = "general") -> str:
    """
    Generate an answer using a small local model.
    Always return a string.
    """
    llm = _get_llm()

    system = {
        "general":    "You are a helpful, brief assistant.",
        "technical":  "You are a precise software assistant. Use short, correct explanations.",
        "educational":"You explain simply with 1–2 short examples.",
        "friendly":   "Be warm and encouraging. Keep answers short.",
    }.get(domain_role, "You are a helpful, brief assistant.")

    instruction = (
        f"{system}\n\n"
        f"Question: {prompt}\n"
        f"Answer briefly:"
    )

    out = llm(
        instruction,
        max_new_tokens=128,
        do_sample=False,
        num_return_sequences=1,
    )[0]["generated_text"]

    return str(out).strip()


def generate_answer(prompt: str, domain_role: str = "general") -> str:
    try:
        response = _local_generate(str(prompt), domain_role=str(domain_role))
        return str(response)
    except Exception as e:
        return f"[LLM error: {type(e).__name__}]"





