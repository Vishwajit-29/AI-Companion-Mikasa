import asyncio
import os
import time
from typing import List

from audio_player import AudioPlayer
from speech_planner import SpeechAction, SpeechPlanner
from tts_provider import TTSProvider

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ConversationState:
    """Manages the global state of the conversation, primarily for interruption."""

    def __init__(self):
        self.generation_id = 0
        self.lock = asyncio.Lock()

    async def new_generation(self):
        """Increments the generation ID to invalidate old audio chunks."""
        async with self.lock:
            self.generation_id += 1
            return self.generation_id


class TTSOrchestrator:
    """
    Orchestrates the entire TTS process:
    - Receives text from the LLM.
    - Uses SpeechPlanner to create a sequence of actions.
    - Executes actions using TTSProvider (for speech) and AudioPlayer (for clips).
    - Handles interruptions via ConversationState.
    """

    def __init__(
        self,
        tts_provider: TTSProvider,
        audio_player: AudioPlayer,
        conversation_state: ConversationState,
    ):
        self.tts_provider = tts_provider
        self.audio_player = audio_player
        self.state = conversation_state
        self.planner = SpeechPlanner()
        self.audio_clips = self._load_audio_clips()

    def _load_audio_clips(self):
        """Loads non-verbal audio clips from the assets directory."""
        clips_path = os.path.join(project_root, "assets", "audio")
        if not os.path.isdir(clips_path):
            print(f"⚠️ Audio clips directory not found: {clips_path}")
            return {}

        loaded_clips = {}
        for filename in os.listdir(clips_path):
            if filename.endswith(".wav"):
                clip_name = os.path.splitext(filename)[0]
                try:
                    with open(os.path.join(clips_path, filename), "rb") as f:
                        loaded_clips[clip_name] = f.read()
                except IOError as e:
                    print(f"⚠️ Failed to load audio clip {filename}: {e}")
        return loaded_clips

    async def process_text(self, text: str):
        """
        Processes a stream of text from the LLM, generating and playing audio.
        """
        current_generation_id = self.state.generation_id
        self.audio_player.start()

        # Feed text to the planner and get actions
        actions = self.planner.feed(text)
        await self._execute_actions(actions, current_generation_id)

    async def finalize(self):
        """Processes any remaining buffered text at the end of a stream."""
        current_generation_id = self.state.generation_id
        actions = self.planner.finalize()
        await self._execute_actions(actions, current_generation_id)
        # Optional: wait for the last audio chunk to finish playing
        self.audio_player.wait_until_done()
        self.audio_player.stop()

    async def _execute_actions(
        self, actions: List[SpeechAction], current_generation_id: int
    ):
        """Executes a list of speech actions."""
        for action in actions:
            # Before each action, check if we've been interrupted
            if self.state.generation_id != current_generation_id:
                print("🛑 Interruption detected, stopping audio playback.")
                self.audio_player.stop()
                return

            if action.type == "speech":
                await self._handle_speech_action(action, current_generation_id)
            elif action.type == "audio":
                self._handle_audio_action(action)
            elif action.type == "pause":
                await asyncio.sleep(action.content / 1000.0)

    async def _handle_speech_action(
        self, action: SpeechAction, current_generation_id: int
    ):
        """Handles a 'speech' action by streaming TTS audio."""
        async for audio_chunk in self.tts_provider.stream(
            action.content, current_generation_id
        ):
            # Check for interruption *again* as chunks arrive
            if audio_chunk.generation_id != self.state.generation_id:
                break  # Stop processing chunks from an old generation
            self.audio_player.play(audio_chunk.pcm)

    def _handle_audio_action(self, action: SpeechAction):
        """Handles an 'audio' action by playing a pre-recorded clip."""
        clip_name = action.content
        if clip_name in self.audio_clips:
            self.audio_player.play(self.audio_clips[clip_name])
        else:
            print(f"⚠️ Audio clip not found: {clip_name}")


# ============================================
# TEST CODE
# ============================================
if __name__ == "__main__":

    async def main_test():
        print("🎵 Testing TTSOrchestrator...")
        print("=" * 50)

        # --- Setup ---
        model = os.path.join(project_root, "tts", "piper", "en_US-amy-medium.onnx")
        piper_exe = os.path.join(project_root, "tts", "piper", "piper.exe")

        try:
            state = ConversationState()
            provider = TTSProvider(model_path=model, piper_path=piper_exe)
            player = AudioPlayer(sample_rate=22050)
            orchestrator = TTSOrchestrator(provider, player, state)
            print("✅ Orchestrator initialized.")
        except FileNotFoundError as e:
            print(f"❌ {e}")
            return

        # --- Test 1: Simple sentence ---
        print("\n--- Test 1: Simple Sentence ---")
        test_text_1 = "Hello, this is a basic test."
        print(f"📝 Input: '{test_text_1}'")
        await orchestrator.process_text(test_text_1)
        await orchestrator.finalize()
        print("✅ Test 1 Complete.")
        time.sleep(1)

        # --- Test 2: Sentence with laughter ---
        print("\n--- Test 2: Laughter ---")
        test_text_2 = "That's pretty funny haha!"
        print(f"📝 Input: '{test_text_2}'")
        await orchestrator.process_text(test_text_2)
        await orchestrator.finalize()
        print("✅ Test 2 Complete.")
        time.sleep(1)

        # --- Test 3: Interruption ---
        print("\n--- Test 3: Interruption ---")
        print("Will start a long sentence, then interrupt after 2 seconds.")
        test_text_3 = "This is a much longer sentence designed to test the interruption functionality of the system, which is critical for a responsive user experience."
        print(f"📝 Input: '{test_text_3}'")

        async def interrupt():
            await asyncio.sleep(2)
            print("\n🚨 --- INTERRUPTING --- 🚨")
            await state.new_generation()
            player.stop()
            print("🚨 --- NEW GENERATION ID SET --- 🚨\n")

        # Run the long sentence and the interruption concurrently
        await asyncio.gather(
            orchestrator.process_text(test_text_3),
            interrupt(),
        )
        await orchestrator.finalize()
        print("✅ Test 3 Complete.")

        # --- Cleanup ---
        provider.stop()
        print("\n🎵 Orchestrator test complete!")

    asyncio.run(main_test())
