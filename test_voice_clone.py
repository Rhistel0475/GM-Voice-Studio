#!/usr/bin/env python3
"""Quick test: load Pocket TTS and generate audio from a preset voice.
Run from project root with HF_TOKEN set (or after hf auth login):
    python test_voice_clone.py
"""
import os
import sys

# Use app's HF token injection so gated model loads
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tts_service import _inject_hf_token

_inject_hf_token()

from pocket_tts import TTSModel
import scipy.io.wavfile

def main():
    print("Loading Pocket TTS (may download gated weights on first run)...")
    tts_model = TTSModel.load_model()
    print("Using built-in preset voice 'alba'...")
    voice_state = tts_model.get_state_for_audio_prompt("alba")
    print("Generating audio...")
    audio = tts_model.generate_audio(voice_state, "Hello world, this is a test.")
    out_path = "output_test.wav"
    scipy.io.wavfile.write(out_path, tts_model.sample_rate, audio.numpy())
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
