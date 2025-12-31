import asyncio
import os

from audio_player import AudioPlayer
from tts_orchestrator import ConversationState, TTSOrchestrator
from tts_provider import TTSProvider

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TTSClient:
    """
    High-level client for the Text-to-Speech system.
    Initializes and manages all TTS components.
    """

    def __init__(self):
        self.state = ConversationState()
        self.audio_player = AudioPlayer(sample_rate=22050)

        try:
            model_path = os.path.join(
                project_root, "tts", "piper", "en_US-amy-medium.onnx"
            )
            piper_path = os.path.join(project_root, "tts", "piper", "piper.exe")

            self.tts_provider = TTSProvider(
                model_path=model_path, piper_path=piper_path
            )
            self.orchestrator = TTSOrchestrator(
                self.tts_provider, self.audio_player, self.state
            )
            self.is_initialized = True
            print("✅ TTS Client initialized successfully.")

        except FileNotFoundError as e:
            print(f"❌ TTS initialization failed: {e}")
            print("👉 TTS will be disabled.")
            self.is_initialized = False
            self.tts_provider = None
            self.orchestrator = None

    async def speak(self, text_stream):
        """
        Primary method to speak text from a streaming source.
        """
        if not self.is_initialized:
            return

        # Start a new generation for this response
        await self.state.new_generation()
        self.audio_player.stop()  # Ensure any previous audio is stopped

        async for text_chunk in text_stream:
            if text_chunk:
                await self.orchestrator.process_text(text_chunk)

        # Finalize the stream to play any remaining buffered text
        await self.orchestrator.finalize()

    def interrupt(self):
        """
        Interrupts any ongoing speech.
        This is called when new user input is detected.
        """
        if not self.is_initialized:
            return

        # This will cause the orchestrator to stop processing old audio
        asyncio.run(self.state.new_generation())
        self.audio_player.stop()
        print("🎤 Speech interrupted.")

    def shutdown(self):
        """Shuts down the TTS provider."""
        if self.tts_provider:
            self.tts_provider.stop()
            print("🛑 TTS provider shut down.")


# ============================================
# TEST CODE
# ============================================
if __name__ == "__main__":

    async def main_test():
        print("🎵 Testing TTSClient...")
        print("=" * 50)

        tts_client = TTSClient()
        if not tts_client.is_initialized:
            print("❌ Test cannot proceed. Exiting.")
            return

        # --- Mock LLM Stream ---
        async def mock_llm_stream(text):
            # Yield text in small, simulated chunks
            words = text.split()
            for i in range(0, len(words), 2):
                yield " ".join(words[i : i + 2]) + " "
                await asyncio.sleep(0.1)  # Simulate network latency

        # --- Test Run ---
        test_sentence = "This is a test of the complete TTS client, streaming from a mock LLM response. It should include pauses, and maybe even some laughter... haha!"
        print(f"\n📝 Input sentence: {test_sentence}")
        print("\n🔊 Calling tts_client.speak()...")

        await tts_client.speak(mock_llm_stream(test_sentence))

        print("\n✅ Speak call finished.")

        # --- Test Interruption ---
        print("\n--- Testing Interruption ---")
        long_sentence = "This is a very long sentence to demonstrate interruption. I will keep talking until the main test function calls the interrupt method, which should stop me mid-sentence."

        async def speak_and_interrupt():
            speak_task = asyncio.create_task(
                tts_client.speak(mock_llm_stream(long_sentence))
            )
            await asyncio.sleep(3)  # Let it speak for 3 seconds
            print("\n🚨 --- INTERRUPTING NOW --- 🚨")
            tts_client.interrupt()
            await speak_task  # Allow the task to finish cleanly

        await speak_and_interrupt()
        print("✅ Interruption test complete.")

        # --- Shutdown ---
        tts_client.shutdown()
        print("\n🎵 TTSClient test complete!")

    asyncio.run(main_test())
