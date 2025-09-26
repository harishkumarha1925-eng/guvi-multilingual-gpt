# src/config/settings.py
from dataclasses import dataclass
import os
from typing import Tuple


@dataclass(frozen=True)
class _Settings:
    # App UI
    APP_TITLE: str = "GUVI Multilingual GPT Chatbot"

    # LLM/runtime
    LLM_MODE: str = os.getenv("LLM_MODE", "local_small")  # local_small | local_large | remote
    DEVICE: str = os.getenv("DEVICE", "cpu")              # cpu | cuda

    # Translation languages (NLLB / FLORES codes)
    # Keep English plus five Indian languages
    SUPPORTED_LANGS: Tuple[str, ...] = (
        "eng_Latn",  # English
        "tam_Taml",  # Tamil
        "hin_Deva",  # Hindi
        "tel_Telu",  # Telugu
        "kan_Knda",  # Kannada
        "mal_Mlym",  # Malayalam
    )

    # Translation defaults
    DEFAULT_TARGET: str = "eng_Latn"   # translate user input into this before LLM
    DETECT_FALLBACK: str = "eng_Latn"  # fallback if language detection is uncertain

    # Generation safeguards
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "512"))
    TIMEOUT_S: int = int(os.getenv("LLM_TIMEOUT_S", "30"))


# Single shared instance imported by the app
settings = _Settings()
