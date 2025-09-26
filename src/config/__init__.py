from types import SimpleNamespace
import os

settings = SimpleNamespace(
    APP_TITLE=os.getenv("APP_TITLE", "Multilingual Mentor"),
    APP_PAGE_ICON=os.getenv("APP_PAGE_ICON", "🧠"),
    APP_ICON=os.getenv("APP_ICON", "🧠"),
    THEME_PRIMARY_COLOR=os.getenv("THEME_PRIMARY_COLOR", "#4F46E5"),

    # HF / model choices (adjust as you like)
    MODEL_NAME=os.getenv("MODEL_NAME", "distilbert-base-multilingual-cased"),

    # 🔧 NEW: this is what your app is trying to read
    LLM_MODE=os.getenv("LLM_MODE", "general"),
)
