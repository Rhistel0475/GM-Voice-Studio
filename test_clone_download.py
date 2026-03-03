#!/usr/bin/env python3
"""Test that the gated voice-cloning model downloads and cloning works.
Run from project root with HF_TOKEN in .env or environment:
    python test_clone_download.py
Creates a short test WAV, loads the model (with cloning weights), then runs clone.
"""
import os
import sys
import tempfile

# Load .env so HF_TOKEN is set before config is imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

from config import HF_TOKEN
from tts_service import _inject_hf_token

def main():
    if not HF_TOKEN:
        print("HF_TOKEN is not set. Set it in .env or: export HF_TOKEN=hf_...")
        print("Also accept terms at https://huggingface.co/kyutai/pocket-tts")
        sys.exit(1)
    print("HF_TOKEN is set. Injecting token and loading Pocket TTS (this may download gated weights)...")
    _inject_hf_token()

    from pocket_tts import TTSModel
    import soundfile as sf
    import numpy as np

    model = TTSModel.load_model()
    if not getattr(model, "has_voice_cloning", True):
        print("FAIL: Model loaded without voice cloning (gated weights did not download).")
        print("Check: 1) HF_TOKEN in .env 2) Accept terms at https://huggingface.co/kyutai/pocket-tts")
        sys.exit(2)
    print("OK: Model has voice cloning.")

    # Create a short test WAV (5 s silence at 24 kHz so clone accepts it)
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        sr = 24000
        duration_sec = 5.0
        samples = int(sr * duration_sec)
        silence = np.zeros(samples, dtype=np.float32)
        sf.write(wav_path, silence, sr)
        print("Cloning from a short test WAV...")
        state = model.get_state_for_audio_prompt(wav_path)
        print("OK: Voice cloning from audio works.")
    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)

    print("All checks passed. Voice cloning should work in the app.")

if __name__ == "__main__":
    main()
