#!/usr/bin/env python3
"""Test that KaniTTS-2 models download and voice cloning works.
Run from project root:
    python test_clone_download.py
No HF token required — models are public.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import soundfile as sf

def main():
    print("Loading KaniTTS-2 (may download models on first run)...")
    from kani_tts import KaniTTS, SpeakerEmbedder
    model = KaniTTS("nineninesix/kani-tts-2-en")
    print("OK: KaniTTS-2 model loaded.")

    # Create a short test WAV (5s silence at 16kHz for SpeakerEmbedder)
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        sr = 16000
        silence = np.zeros(int(sr * 5.0), dtype=np.float32)
        sf.write(wav_path, silence, sr)
        print("Cloning from a short test WAV...")
        embedder = SpeakerEmbedder(model_name="nineninesix/speaker-emb-tbr", max_duration_sec=30.0)
        emb = embedder.embed_audio_file(wav_path)
        print(f"OK: Speaker embedding shape: {emb.shape}")
    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)

    print("All checks passed. Voice cloning should work in the app.")

if __name__ == "__main__":
    main()
