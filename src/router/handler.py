def handle_turn(user_text: str) -> str:
    if not user_text:
        return "Say hi 👋"
    return f"Echo: {user_text}"
