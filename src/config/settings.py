# src/config/settings.py
from dataclasses import dataclass

@dataclass
class Settings:
    device: str = "cpu"

settings = Settings()
