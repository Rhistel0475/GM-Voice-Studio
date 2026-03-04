"""
TTS service: thin interface over KaniTTS-2 (nineninesix.ai).
Callers get (audio_array, sample_rate). English-only; supports cloned voices (.pt tensors).
No preset named voices — pass speaker_emb_path=None for a random voice.
"""
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch

from config import AUDIO_CACHE_SIZE

KANI_SAMPLE_RATE = 22050
DEFAULT_LANGUAGE_TAGS = ["en"]

_model = None
_audio_cache: list[str] = []


def _get_tts():
    global _model
    if _model is None:
        from kani_tts import KaniTTS
        logging.info("Loading KaniTTS-2...")
        _model = KaniTTS("nineninesix/kani-tts-2-en")
    return _model


def is_model_loaded() -> bool:
    return _model is not None


def get_supported_language_tags() -> list[str]:
    return list(DEFAULT_LANGUAGE_TAGS)


def get_preset_voices() -> list[str]:
    """KaniTTS-2 has no named preset voices; returns empty list."""
    return []


def _is_preset_voice(voice_id: str) -> bool:
    """KaniTTS-2 has no named preset voices; always returns False."""
    return False


def _evict_old_audio():
    while len(_audio_cache) >= AUDIO_CACHE_SIZE and _audio_cache:
        path = _audio_cache.pop(0)
        try:
            os.unlink(path)
        except OSError:
            pass


def generate(
    text: str,
    language_tag: Optional[str] = "en",
    speaker_emb_path: Optional[str] = None,
    temperature: float = 1.0,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
) -> tuple[np.ndarray, int]:
    """Generate speech.
    speaker_emb_path: path to a .pt speaker embedding file produced by clone_voice().
    Pass None for a random/default voice. language_tag is ignored (English only).
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("Text is required")

    if speaker_emb_path and speaker_emb_path.strip():
        p = Path(speaker_emb_path.strip())
        if not p.exists():
            raise ValueError("Voice not found. Select a cloned voice or leave voice unset for random.")
        emb = torch.load(str(p), weights_only=True)
    else:
        emb = None

    model = _get_tts()
    try:
        kwargs = dict(temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty)
        if emb is not None:
            audio, _out_text = model(text, speaker_emb=emb, **kwargs)
        else:
            audio, _out_text = model(text, **kwargs)
        arr = np.array(audio) if not isinstance(audio, np.ndarray) else audio
    except Exception as e:
        logging.exception("TTS generate failed")
        raise RuntimeError(f"Generation failed: {e!s}") from e

    return arr, KANI_SAMPLE_RATE


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
