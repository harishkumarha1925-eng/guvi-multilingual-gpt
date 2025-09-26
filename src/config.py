from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseModel):
    LLM_MODE: str = os.getenv("LLM_MODE", "local_small")
    HF_API_TOKEN: str | None = os.getenv("HF_API_TOKEN")
    HF_TEXT_GENERATION_MODEL: str = os.getenv("HF_TEXT_GENERATION_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    NLLB_MODEL: str = os.getenv("NLLB_MODEL", "facebook/nllb-200-distilled-600M")
    NLLB_TGT_LANG: str = os.getenv("NLLB_TGT_LANG", "eng_Latn")
    APP_TITLE: str = os.getenv("APP_TITLE", "GUVI Multilingual GPT Chatbot")

settings = Settings()
