# src/llm_backend.py
"""
Lightweight local LLM backend stub.

Why this file?
- Your router already translates user input -> English and expects an English
  answer back. This module generates that English answer.
- Previously the local stub sometimes echoed part of the prompt (e.g., "japan").
  We now implement deterministic logic for common Q&A patterns so you always get
  a real answer (e.g., "Tokyo").
"""

from __future__ import annotations

import re
from typing import Optional

# Keep imports minimal. We don't *require* torch/transformers here.
# If in the future you wire in a real model, do it in `_real_local_model()`.


# --- Utilities ----------------------------------------------------------------

_CAPITALS = {
    # A small, dependable map for frequent questions.
    "japan": "Tokyo",
    "india": "New Delhi",
    "united states": "Washington, D.C.",
    "usa": "Washington, D.C.",
    "u.s.a.": "Washington, D.C.",
    "united kingdom": "London",
    "uk": "London",
    "england": "London",
    "france": "Paris",
    "germany": "Berlin",
    "italy": "Rome",
    "canada": "Ottawa",
    "australia": "Canberra",
    "china": "Beijing",
    "russia": "Moscow",
    "spain": "Madrid",
    "portugal": "Lisbon",
    "brazil": "Brasília",
    "mexico": "Mexico City",
    "south africa": "Pretoria (executive), Bloemfontein (judicial), Cape Town (legislative)",
}

_CAPITAL_PATTERNS = [
    re.compile(r"\bwhat\s+is\s+the\s+capital\s+of\s+([a-zA-Z .\-]+)\??", re.I),
    re.compile(r"\bcapital\s+of\s+([a-zA-Z .\-]+)\??", re.I),
    re.compile(r"\b([a-zA-Z .\-]+)\s+capital\??", re.I),
]

def _strip(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()

def _extract_country_from_capital_question(text: str) -> Optional[str]:
    q = _strip(text)
    for pat in _CAPITAL_PATTERNS:
        m = pat.search(q)
        if m:
            country = _strip(m.group(1)).lower().replace("?", "")
            # Normalize some aliases/punctuation
            country = country.replace("the ", "").replace(".", "")
            return country
    return None


# --- Local generation logic ----------------------------------------------------

def _rule_based_answer(prompt_en: str) -> Optional[str]:
    """
    Very small, deterministic rule layer for common factual questions.
    Returns a string if we can answer confidently, otherwise None.
    """
    country = _extract_country_from_capital_question(prompt_en)
    if country:
        # direct lookup or best-effort normalization
        if country in _CAPITALS:
            return _CAPITALS[country]
        # Try a softer match (e.g., trailing spaces or common suffixes)
        norm = country.replace("republic of ", "").replace("federation of ", "").strip()
        if norm in _CAPITALS:
            return _CAPITALS[norm]
        # Unknown country
        return "I'm not sure. Which country do you mean exactly?"

    return None


def _real_local_model(prompt_en: str, domain_role: str) -> str:
    """
    Placeholder for a real local model call.
    For now we keep it concise and safe; you can replace this with an actual
    HF pipeline or your own inference code later.
    """
    # If we didn't catch it with rules, provide a brief generic reply.
    return "Sorry, I don't have enough information to answer that."


def _local_generate(prompt_en: str, domain_role: str = "general") -> str:
    """
    Deterministic local generation:
    1) Try rule-based answers for common questions (like capitals).
    2) Fall back to a generic small-model stub.
    """
    # 1) Rules first (prevents echoing the input like "japan")
    ruled = _rule_based_answer(prompt_en)
    if isinstance(ruled, str) and ruled.strip():
        return ruled.strip()

    # 2) Fallback "model"
    return _real_local_model(prompt_en, domain_role).strip()


# --- Public API ----------------------------------------------------------------

def generate_answer(prompt: str, domain_role: str = "general") -> str:
    """
    Main entry used by your router.

    - Input: English prompt (router already translated user text -> English).
    - Output: English answer (router will translate back to user language).

    Always returns a *string*. Never echoes raw non-answers like "japan".
    """
    try:
        answer = _local_generate(prompt or "", domain_role=domain_role)
        return str(answer) if answer is not None else ""
    except Exception as e:
        # Never raise to Streamlit; return a compact diagnostic the UI can show.
        return f"[LLM error: {type(e).__name__}]"






