# generate_placeholders.py
import numpy as np
import soundfile as sf
import os

os.makedirs("assets/audio", exist_ok=True)

sample_rate = 22050

# Short silence (300ms) as placeholder
silence = np.zeros(int(sample_rate * 0.3), dtype=np.float32)
sf.write("assets/audio/laugh_soft.wav", silence, sample_rate)
sf.write("assets/audio/laugh_short.wav", silence, sample_rate)

print("✅ Placeholder audio files created!")
