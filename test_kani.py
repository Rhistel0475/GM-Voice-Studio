# Pocket TTS smoke test.
# Run with: python test_kani.py
from pocket_tts import TTSModel

model = TTSModel.load_model()
print("Pocket TTS model loaded OK")
