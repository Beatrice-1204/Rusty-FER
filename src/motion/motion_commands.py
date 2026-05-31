EMOTION_COMMANDS = {
    "happy": "H",
    "angry": "G",
    "sad": "V",
    "surprise": "U",
    "neutral": None,
}
 
STOP_COMMAND = "x"


def command_for_emotion(emotion: str) -> str | None:
    return EMOTION_COMMANDS.get(emotion.lower())
