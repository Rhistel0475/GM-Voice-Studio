"""
TTS service: thin interface over the Kani engine.
Callers get (audio_array, sample_rate); engine can be swapped later.
"""
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from config import AUDIO_CACHE_SIZE, MODEL_NAME
from kani_tts import KaniTTS

# Fallback when model not loaded or has no language_tags_list
DEFAULT_LANGUAGE_TAGS = ["en_us", "en_nyork", "en_oakl", "en_glasg", "en_bost", "en_scou"]

# Load once at startup
_tts: Optional[KaniTTS] = None
_audio_cache: list[str] = []


def _get_tts() -> KaniTTS:
    global _tts
    if _tts is None:
        _tts = KaniTTS(MODEL_NAME)
    return _tts


def is_model_loaded() -> bool:
    """True if the TTS model has been loaded at least once (for readiness probe)."""
    return _tts is not None


def get_supported_language_tags() -> list[str]:
    """
    Return the list of preset accent tags supported by the loaded model.
    Uses the model's language_tags_list when available; otherwise DEFAULT_LANGUAGE_TAGS.
    """
    try:
        tts = _get_tts()
        if getattr(tts, "language_tags_list", None):
            return list(tts.language_tags_list)
    except Exception:
        pass
    return list(DEFAULT_LANGUAGE_TAGS)


def _evict_old_audio():
    while len(_audio_cache) >= AUDIO_CACHE_SIZE and _audio_cache:
        path = _audio_cache.pop(0)
        try:
            os.unlink(path)
        except OSError:
            pass


def generate(
    text: str,
    language_tag: Optional[str] = None,
    speaker_emb_path: Optional[str] = None,
    temperature: float = 1.0,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
) -> tuple[np.ndarray, int]:
    """
    Generate speech from text. Thin interface so the engine can be swapped.

    Args:
        text: Input text (non-empty).
        language_tag: Optional preset voice/accent (e.g. en_us, en_nyork).
        speaker_emb_path: Optional path to .pt speaker embedding for custom voice.
        temperature: Sampling temperature.
        top_p: Top-p sampling.
        repetition_penalty: Repetition penalty.

    Returns:
        (audio_ndarray, sample_rate).

    Raises:
        ValueError: If text is empty.
        RuntimeError: On TTS failure (logged).
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("Text is required")

    tts = _get_tts()
    kwargs: dict = {
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
    }
    if language_tag:
        kwargs["language_tag"] = language_tag
    if speaker_emb_path:
        path = Path(speaker_emb_path)
        if not path.exists():
            raise FileNotFoundError(f"Speaker embedding not found: {speaker_emb_path}")
        kwargs["speaker_emb"] = str(path)

    try:
        audio, _ = tts(text, **kwargs)
    except Exception as e:
        logging.exception("TTS generate failed")
        raise RuntimeError(f"Generation failed: {e!s}") from e

    return audio, tts.sample_rate


def generate_to_file(
    text: str,
    language_tag: Optional[str] = None,
    speaker_emb_path: Optional[str] = None,
) -> str:
    """
    Generate speech and write to a temp WAV file. Uses bounded cache.
    For Gradio or any caller that needs a file path.
    """
    audio, sample_rate = generate(text, language_tag=language_tag, speaker_emb_path=speaker_emb_path)
    _evict_old_audio()
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        tts = _get_tts()
        tts.save_audio(audio, path)
    except Exception as e:
        try:
            os.unlink(path)
        except OSError:
            pass
        logging.exception("save_audio failed")
        raise RuntimeError(f"Could not save audio: {e!s}") from e
    _audio_cache.append(path)
    return path
