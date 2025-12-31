"""
Direct Piper test - bypasses streaming to diagnose the issue
"""

import os
import subprocess


def test_piper_direct():
    """Test Piper by generating a WAV file first"""

    piper_path = "tts/piper/piper.exe"
    model_path = "tts/piper/en_US-amy-medium.onnx"
    output_wav = "test_output.wav"

    print("🧪 PIPER DIAGNOSTIC TEST")
    print("=" * 50)

    # Check files exist
    if not os.path.exists(piper_path):
        print(f"❌ Piper not found at: {piper_path}")
        return

    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        return

    print(f"✅ Piper found: {piper_path}")
    print(f"✅ Model found: {model_path}")
    print()

    # Test 1: Generate WAV file
    print("📝 Test 1: Generating WAV file...")
    test_text = "Hello, this is a test."

    cmd = [piper_path, "--model", model_path, "--output_file", output_wav]

    try:
        result = subprocess.run(
            cmd, input=test_text.encode("utf-8"), capture_output=True, timeout=10
        )

        if result.returncode != 0:
            print(f"❌ Piper failed with error:")
            print(result.stderr.decode())
            return

        if os.path.exists(output_wav):
            size = os.path.getsize(output_wav)
            print(f"✅ WAV file created: {output_wav} ({size} bytes)")
            print()
            print("▶️  Now play this file manually to verify Piper works:")
            print(f"   Double-click: {output_wav}")
            print()
        else:
            print("❌ No output file created")
            return

    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # Test 2: Try raw output
    print("📝 Test 2: Testing raw PCM output...")

    cmd_raw = [piper_path, "--model", model_path, "--output-raw"]

    try:
        result = subprocess.run(
            cmd_raw, input=test_text.encode("utf-8"), capture_output=True, timeout=10
        )

        pcm_data = result.stdout

        if len(pcm_data) > 0:
            print(f"✅ Raw PCM generated: {len(pcm_data)} bytes")
            print()

            # Now try to play it
            print("📢 Test 3: Attempting to play raw PCM...")

            import numpy as np
            import sounddevice as sd

            # Convert to audio
            audio_data = np.frombuffer(pcm_data, dtype=np.int16)
            audio_float = audio_data.astype(np.float32) / 32768.0

            print("▶️  Playing...")
            sd.play(audio_float, samplerate=22050, blocking=True)
            sd.wait()

            print("✅ Playback complete!")
            print()
            print("🎉 SUCCESS! Piper is working correctly.")
            print("   The issue was in the streaming setup.")

        else:
            print("❌ No PCM data received")
            print("Stderr:", result.stderr.decode())

    except Exception as e:
        print(f"❌ Error during PCM test: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_piper_direct()
