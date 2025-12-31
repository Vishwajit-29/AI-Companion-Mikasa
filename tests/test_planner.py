import os
import sys

# Ensure project root is on sys.path so `tts` can be imported when running this script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tts.speech_planner import SpeechPlanner

planner = SpeechPlanner()


# Simulate streaming tokens
test_tokens = [
    "Yeah",
    ",",
    " ",
    "haha",
    " ",
    "that's",
    " ",
    "actually",
    " ",
    "pretty",
    " ",
    "funny",
    ".",
    " ",
    "I",
    " ",
    "didn't",
    " ",
    "expect",
    " ",
    "that",
    "!",
]

print("🧠 Testing Speech Planner\n")
print("Tokens:", "".join(test_tokens))
print("\n" + "=" * 60 + "\n")

for token in test_tokens:
    actions = planner.feed(token)
    if actions:
        for action in actions:
            if action.type == "speech":
                print(f'🗣️  SPEAK: "{action.content}"')
            elif action.type == "pause":
                print(f"⏸️  PAUSE: {action.content}ms")
            elif action.type == "audio":
                print(f"🎵 AUDIO: {action.content}.wav")

# Finalize any remaining buffer
final_actions = planner.finalize()
for action in final_actions:
    if action.type == "speech":
        print(f'🗣️  SPEAK: "{action.content}"')
