# src/config/settings.py

from __future__ import annotations
import os
from pydantic import BaseModel
from dotenv import load_dotenv

# Load .env file (if exists)
load_dotenv()


class Settings(BaseModel):
    # ---- LLM Config ----
    LLM_MODE: str = os.getenv("LLM_MODE", "local_small")  # local_small | hf_inference
    HF_API_TOKEN: str | None = os.getenv("HF_API_TOKEN")
    HF_TEXT_GENERATION_MODEL: str = os.getenv(
        "HF_TEXT_GENERATION_MODEL", "google/flan-t5-small"
    )

    # ---- Translation ----
    NLLB_MODEL: str = os.getenv("NLLB_MODEL", "facebook/nllb-200-distilled-600M")
    NLLB_SRC_LANG: str = os.getenv("NLLB_SRC_LANG", "auto")
    NLLB_TGT_LANG: str = os.getenv("NLLB_TGT_LANG", "eng_Latn")

    # ---- App UI ----
    APP_TITLE: str = os.getenv("APP_TITLE", "GUVI Multilingual GPT Chatbot")


# Instantiate global settings object
settings = Settings()
