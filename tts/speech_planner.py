import re
import time
from dataclasses import dataclass
from typing import List, Literal


@dataclass
class SpeechAction:
    """Represents a single speech action"""

    type: Literal["pause", "speech", "audio"]
    content: str | int  # text for speech, ms for pause, filename for audio


class SpeechPlanner:
    """
    Converts streaming LLM tokens into natural speech actions.
    Adds pauses, detects laughter, handles punctuation timing.
    """

    def __init__(self):
        self.buffer = ""
        self.actions = []

    def feed(self, token: str) -> List[SpeechAction]:
        """
        Feed a token from LLM, get back speech actions (if ready).
        Returns actions when a natural break is detected.
        """
        self.buffer += token

        # Check if we should flush (sentence end, pause, etc.)
        if self._should_flush():
            return self._flush()

        return []

    def _should_flush(self) -> bool:
        """Decide if buffer is ready to convert to speech"""
        # Flush on newlines
        if "\n" in self.buffer:
            return True

        # Flush if buffer gets too long (prevent lag)
        if len(self.buffer) > 300:
            return True

        return False

    def _flush(self) -> List[SpeechAction]:
        """Convert buffered text to speech actions"""
        actions = []

        # Split by newlines and process each line
        lines = self.buffer.split("\n")
        self.buffer = lines.pop()  # Keep the last partial line in the buffer

        for text in lines:
            text = text.strip()
            if not text:
                continue

            # Detect laughter patterns
            laughter_patterns = [
                (r"\b(haha|hahaha)\b", "laugh_short_1"),
                (r"\b(lol|lmao)\b", "laugh_soft_1"),
                (r"😂|🤣", "laugh_short_1"),
            ]

            laughter_found = False
            for pattern, audio_clip in laughter_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    # Split around laughter
                    parts = re.split(f"({pattern})", text, flags=re.IGNORECASE)

                    for part in parts:
                        part = part.strip()
                        if not part:
                            continue

                        if re.match(pattern, part, re.IGNORECASE):
                            # It's laughter - add audio clip
                            actions.append(SpeechAction("audio", audio_clip))
                        else:
                            # Regular text - add speech
                            actions.append(SpeechAction("speech", part))

                    # Add pause after laughter
                    actions.append(SpeechAction("pause", 150))
                    laughter_found = True
                    break

            if laughter_found:
                continue

            # No laughter detected - handle punctuation pauses

            # Sentence ending (. ! ?)
            if re.search(r"[.!?]\s*$", text):
                actions.append(SpeechAction("speech", text))
                actions.append(SpeechAction("pause", 300))  # Longer pause

            # Comma pause
            elif re.search(r",\s*$", text):
                actions.append(SpeechAction("speech", text))
                actions.append(SpeechAction("pause", 120))  # Short pause

            # Ellipsis (thinking pause)
            elif "..." in text:
                actions.append(SpeechAction("speech", text))
                actions.append(SpeechAction("pause", 400))  # Longer thinking pause

            # No special punctuation
            else:
                actions.append(SpeechAction("speech", text))

        return actions

    def finalize(self) -> List[SpeechAction]:
        """Force flush any remaining buffer (end of response)"""
        if self.buffer.strip():
            text = self.buffer.strip()
            self.buffer = ""
            return [SpeechAction("speech", text)]
        return []
