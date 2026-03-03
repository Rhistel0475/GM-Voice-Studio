"""
TTS service: thin interface over Pocket TTS (Kyutai).
Callers get (audio_array, sample_rate). English-only; supports preset voices and cloned voices (.safetensors).
"""
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from config import AUDIO_CACHE_SIZE, HF_TOKEN

# Pocket TTS: English only; preset voice names from Kyutai
DEFAULT_LANGUAGE_TAGS = ["en"]
POCKET_PRESET_VOICES = ["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"]

_model = None
_audio_cache: list[str] = []


def _get_tts():
    global _model
    if _model is None:
        # So gated models (e.g. voice cloning) can be downloaded
        if HF_TOKEN:
            os.environ["HF_TOKEN"] = HF_TOKEN
            os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
        # Ensure ALL hf_hub_download calls get our token (Pocket TTS doesn't pass it).
        _inject_hf_token()
        from pocket_tts import TTSModel
        logging.info("Loading Pocket TTS...")
        _model = TTSModel.load_model()
    return _model


def _inject_hf_token():
    """Make huggingface_hub use our token for every download (required for gated kyutai/pocket-tts)."""
    import huggingface_hub.hub_mixin as hub_mixin
    from huggingface_hub import hf_hub_download as _real_hf_hub_download

    token = HF_TOKEN or os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        logging.warning("HF_TOKEN not set; voice cloning download may fail for gated repos.")

    def _hf_hub_download(*args, **kwargs):
        # Inject token so gated repos (e.g. kyutai/pocket-tts) work
        if token and (kwargs.get("token") is None or kwargs.get("token") is False):
            kwargs["token"] = token
        elif not token and "token" not in kwargs:
            kwargs["token"] = True  # use cached CLI token
        return _real_hf_hub_download(*args, **kwargs)

    hub_mixin.hf_hub_download = _hf_hub_download
    # pocket_tts.utils.utils does "from huggingface_hub import hf_hub_download"
    # so we must patch the name they'll import (the module's binding)
    import huggingface_hub
    huggingface_hub.hf_hub_download = _hf_hub_download


def is_model_loaded() -> bool:
    return _model is not None


def get_supported_language_tags() -> list[str]:
    return list(DEFAULT_LANGUAGE_TAGS)


def get_preset_voices() -> list[str]:
    return list(POCKET_PRESET_VOICES)


def _evict_old_audio():
    while len(_audio_cache) >= AUDIO_CACHE_SIZE and _audio_cache:
        path = _audio_cache.pop(0)
        try:
            os.unlink(path)
        except OSError:
            pass


def _is_preset_voice(voice_id: str) -> bool:
    return voice_id and voice_id.strip().lower() in [v.lower() for v in POCKET_PRESET_VOICES]


def generate(
    text: str,
    language_tag: Optional[str] = "en",
    speaker_emb_path: Optional[str] = None,
    temperature: float = 0.65,
    top_p: float = 0.80,
    repetition_penalty: float = 1.15,
) -> tuple[np.ndarray, int]:
    """Generate speech. speaker_emb_path: path to .safetensors file or preset voice name (e.g. alba). language_tag ignored (English only)."""
    text = (text or "").strip()
    if not text:
        raise ValueError("Text is required")
    if not speaker_emb_path or not speaker_emb_path.strip():
        raise ValueError("Pocket TTS requires a voice to be selected (preset or cloned).")

    model = _get_tts()
    voice_ref = speaker_emb_path.strip()
    # Preset name or path to .safetensors (or any path Pocket accepts)
    if not _is_preset_voice(voice_ref) and not Path(voice_ref).exists():
        raise ValueError("Voice not found. Select a built-in voice or a cloned voice.")

    try:
        voice_state = model.get_state_for_audio_prompt(voice_ref)
        audio = model.generate_audio(voice_state, text)
        if hasattr(audio, "numpy"):
            arr = audio.numpy()
        else:
            arr = np.array(audio.cpu())
        sr = model.sample_rate
    except Exception as e:
        logging.exception("TTS generate failed")
        raise RuntimeError(f"Generation failed: {e!s}") from e

    return arr, sr


def generate_to_file(
    text: str,
    language_tag: Optional[str] = "en",
    speaker_emb_path: Optional[str] = None,
) -> str:
    audio, sample_rate = generate(text, language_tag=language_tag, speaker_emb_path=speaker_emb_path)
    _evict_old_audio()
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        sf.write(path, audio, sample_rate)
    except Exception as e:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise RuntimeError(f"Could not save audio: {e!s}") from e
    _audio_cache.append(path)
    return path
