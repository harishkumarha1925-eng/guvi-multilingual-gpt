# src/heuristics.py
from __future__ import annotations
import re
from datetime import datetime

TIME_PAT = re.compile(r"\b(time|clock)\b", re.I)
DATE_PAT = re.compile(r"\b(date|day|today)\b", re.I)
HELLO_PAT = re.compile(r"\b(hi|hello|hey)\b", re.I)

def maybe_answer_locally(user_text_en: str) -> str | None:
    t = user_text_en.strip()
    if not t:
        return None

    if HELLO_PAT.search(t):
        return "Hello! How can I help you today?"

    if TIME_PAT.search(t):
        now = datetime.now().strftime("%I:%M %p")
        return f"The current time is {now}."

    if DATE_PAT.search(t):
        today = datetime.now().strftime("%A, %d %B %Y")
        return f"Today is {today}."

    return None
