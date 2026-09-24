from phase5_tts import TextToSpeech
import os


DESCRIPTION_FILE = "final_description.txt"


if not os.path.exists(DESCRIPTION_FILE):
    print("final_description.txt not found.")
    print("Run phase4_combined.py first.")
    exit()


with open(
    DESCRIPTION_FILE,
    "r",
    encoding="utf-8"
) as f:
    description = f.read().strip()


if not description:
    print("Description is empty.")
    exit()


print("\nSpeaking:")
print(description)


tts = TextToSpeech()
tts.speak(description)