import queue
import threading
import time

import numpy as np
import sounddevice as sd


class AudioPlayer:
    """
    Interrupt-safe audio player for TTS.
    Uses sounddevice's streaming callback for gap-free playback.
    """

    def __init__(self, sample_rate=22050, channels=1, blocksize=4410):
        self.sample_rate = sample_rate
        self.channels = channels
        self.blocksize = blocksize  # samples per callback (~200ms at 22050Hz)
        self.is_playing = False
        self.audio_queue = queue.Queue()
        self.stream = None
        self.lock = threading.Lock()
        self.buffer = np.array([], dtype=np.float32)

    def _audio_callback(self, outdata, frames, time_info, status):
        """
        Callback function called by sounddevice for each audio block.
        This ensures gap-free playback.
        """
        if status:
            print(f"⚠️ Audio status: {status}")

        # Try to get more data from queue
        while len(self.buffer) < frames:
            try:
                pcm_chunk = self.audio_queue.get_nowait()
                # Convert to float32
                audio_data = np.frombuffer(pcm_chunk, dtype=np.int16)
                audio_float = audio_data.astype(np.float32) / 32768.0
                self.buffer = np.concatenate([self.buffer, audio_float])
            except queue.Empty:
                break

        # Fill output with buffered data
        if len(self.buffer) >= frames:
            outdata[:, 0] = self.buffer[:frames]
            self.buffer = self.buffer[frames:]
        else:
            # Not enough data - fill with what we have + silence
            if len(self.buffer) > 0:
                outdata[: len(self.buffer), 0] = self.buffer
                outdata[len(self.buffer) :, 0] = 0
                self.buffer = np.array([], dtype=np.float32)
            else:
                outdata[:, 0] = 0  # Silence

    def play(self, pcm_chunk: bytes):
        """
        Queue a chunk of raw PCM audio for playback.
        Non-blocking - returns immediately.
        """
        if self.is_playing:
            self.audio_queue.put(pcm_chunk)

    def start(self):
        """Start the audio player with streaming callback"""
        with self.lock:
            if self.is_playing:
                return

            self.is_playing = True
            self.buffer = np.array([], dtype=np.float32)

            # Clear queue
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break

            # Start audio stream with callback
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=self._audio_callback,
                blocksize=self.blocksize,
                dtype="float32",
            )
            self.stream.start()

    def stop(self):
        """Stop audio immediately and clear queue"""
        with self.lock:
            self.is_playing = False

            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None

            # Clear queue and buffer
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break

            self.buffer = np.array([], dtype=np.float32)

    def is_active(self):
        """Check if player is currently active"""
        return self.is_playing

    def wait_until_done(self, timeout=10):
        """Wait until all queued audio has been played"""
        start = time.time()
        while not self.audio_queue.empty() or len(self.buffer) > 0:
            if time.time() - start > timeout:
                break
            time.sleep(0.05)
        time.sleep(0.2)  # Extra buffer for last samples


# ============================================
# TEST CODE
# ============================================

if __name__ == "__main__":
    import subprocess

    print("🎵 Testing AudioPlayer with callback streaming...")
    print("=" * 60)

    # Generate audio with Piper
    piper_path = "tts/piper/piper.exe"
    model_path = "tts/piper/en_US-amy-medium.onnx"
    test_text = (
        "Hello! This is a streaming audio test. It should sound smooth and natural."
    )

    print(f"📝 Text: {test_text}")
    print("🔊 Generating audio with Piper...")

    cmd = [piper_path, "--model", model_path, "--output-raw"]
    result = subprocess.run(
        cmd, input=test_text.encode("utf-8"), capture_output=True, timeout=10
    )
    pcm_data = result.stdout

    print(f"✅ Generated {len(pcm_data)} bytes of audio")
    print("\n▶️  Playing with callback streaming...")

    # Test chunked playback
    player = AudioPlayer(sample_rate=22050)
    player.start()

    chunk_size = 8820  # ~200ms chunks
    chunk_count = 0

    try:
        for i in range(0, len(pcm_data), chunk_size):
            chunk = pcm_data[i : i + chunk_size]
            player.play(chunk)
            chunk_count += 1

            if chunk_count <= 5:
                print(f"   📦 Queued chunk {chunk_count}")

        print(f"\n✅ All {chunk_count} chunks queued")
        print("⏳ Waiting for playback to finish...")

        player.wait_until_done()

        print("✅ Playback complete!")

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted!")

    finally:
        player.stop()
        print("🎵 Test complete!")
