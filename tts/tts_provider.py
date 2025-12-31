import os
import subprocess
import threading
from dataclasses import dataclass
from typing import AsyncGenerator

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class AudioChunk:
    """Represents a chunk of audio data with its generation ID."""

    pcm: bytes
    generation_id: int


class TTSProvider:
    """
    Minimalist, streaming-first TTS provider for Piper.
    - Spawns Piper as a subprocess
    - Sends text chunks to stdin
    - Reads raw PCM audio from stdout
    - Yields audio chunks with generation ID
    """

    def __init__(self, model_path: str, piper_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"TTS model not found at: {model_path}")
        if not os.path.exists(piper_path):
            raise FileNotFoundError(f"Piper executable not found at: {piper_path}")

        self.model_path = model_path
        self.piper_path = piper_path
        self.process = None
        self.lock = threading.Lock()

    def _start_process(self):
        """Starts the Piper subprocess."""
        if self.process and self.process.poll() is None:
            return  # Process already running

        cmd = [self.piper_path, "--model", self.model_path, "--output-raw"]
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.path.join(project_root, "tts", "piper"),
        )

    async def stream(
        self, text_chunk: str, generation_id: int
    ) -> AsyncGenerator[AudioChunk, None]:
        """
        Streams audio for a given text chunk.
        Restarts Piper if it crashes.
        """
        with self.lock:
            self._start_process()

            if self.process.poll() is not None:
                # If process died, try restarting it once
                self._start_process()
                if self.process.poll() is not None:
                    print("🚨 Piper process failed to start.")
                    return

            try:
                # Send text to Piper's stdin
                self.process.stdin.write((text_chunk + "\n").encode("utf-8"))
                self.process.stdin.flush()

                # Read audio from stdout
                # This is a blocking read, but Piper sends audio as it's generated
                # In a real async app, we'd use asyncio.subprocess
                while True:
                    # Heuristic: read based on typical audio chunk size
                    # This is not perfect and can introduce latency.
                    # A better approach involves more complex stdout handling.
                    output = self.process.stdout.read(4096)
                    if not output:
                        break
                    yield AudioChunk(pcm=output, generation_id=generation_id)

            except (BrokenPipeError, OSError) as e:
                print(f"🚨 Piper process error: {e}")
                self.process.kill()
                self.process = None
                # Don't yield anything on error

    def stop(self):
        """Stops the Piper subprocess."""
        with self.lock:
            if self.process and self.process.poll() is None:
                self.process.kill()
                self.process = None


# ============================================
# TEST CODE
# ============================================
if __name__ == "__main__":
    import asyncio

    from audio_player import AudioPlayer

    async def main_test():
        print("🎵 Testing TTSProvider...")
        print("=" * 50)

        model = os.path.join(project_root, "tts", "piper", "en_US-amy-medium.onnx")
        piper_exe = os.path.join(project_root, "tts", "piper", "piper.exe")

        player = AudioPlayer(sample_rate=22050)

        try:
            tts_provider = TTSProvider(model_path=model, piper_path=piper_exe)
            print("✅ TTSProvider initialized.")
        except FileNotFoundError as e:
            print(f"❌ {e}")
            print("👉 Please ensure piper.exe and the model are in tts/piper/")
            return

        test_text = "Hello, this is a test of the streaming TTS system."
        print(f"📝 Text: '{test_text}'")
        print("🔊 Streaming and playing audio...")

        try:
            player.start()
            audio_chunks = []
            async for chunk in tts_provider.stream(test_text, generation_id=1):
                audio_chunks.append(chunk.pcm)
                player.play(chunk.pcm)
                print(f"   📦 Received and played audio chunk, size: {len(chunk.pcm)}")

            if not audio_chunks:
                print("❌ No audio received. Piper might have failed.")
            else:
                print(f"✅ Successfully streamed {len(audio_chunks)} chunks.")

            player.wait_until_done()

        except Exception as e:
            print(f"❌ An error occurred during streaming: {e}")
        finally:
            player.stop()
            tts_provider.stop()
            print("🛑 TTSProvider and AudioPlayer stopped.")
            print("\n🎵 Test complete!")

    asyncio.run(main_test())
