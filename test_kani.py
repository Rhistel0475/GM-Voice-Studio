# XTTSv2 smoke test (replaces former Kani TTS script).
# Run with: COQUI_TOS_AGREED=1 python test_kani.py
import os
os.environ.setdefault("COQUI_TOS_AGREED", "1")
from TTS.api import TTS
model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda" if __import__("torch").cuda.is_available() else "cpu")
print("XTTSv2 model loaded OK")
