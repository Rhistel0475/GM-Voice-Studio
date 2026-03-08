#!/usr/bin/env python3
"""Quick test: load KaniTTS-2 and generate audio with a random voice.
Run from project root:
    python test_voice_clone.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kani_tts import KaniTTS
import soundfile as sf

def main():
    print("Loading KaniTTS-2 (may download models on first run)...")
    model = KaniTTS("nineninesix/kani-tts-2-en")
    print("Generating audio with random voice...")
    audio, _text = model("Hello world, this is a test.", language_tag="en_us")
    out_path = "output_test.wav"
    sf.write(out_path, audio, 22050)
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
