# src/llm_backend.py
from typing import Any

# import your local model / pipeline here
# from ._local_model import generate as _local_generate

def _as_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for k in ("generated_text", "text", "response", "answer", "output"):
            v = x.get(k)
            if isinstance(v, str):
                return v
    if isinstance(x, (list, tuple)) and x:
        return _as_text(x[0])
    return str(x)

def _local_generate(prompt: str, domain_role: str = "general") -> Any:
    # ---- replace with your actual local generator ----
    # Example minimal stub so the app runs:
    return f"(role={domain_role}) {prompt}"
    # real world: return pipeline(prompt) / model.generate(...)

def generate_answer(prompt: str, domain_role: str = "general") -> str:
    try:
        raw = _local_generate(prompt, domain_role=domain_role)
        return _as_text(raw).strip()
    except Exception as e:
        return f"[LLM error: {type(e).__name__}]"




