"""
Voice clone pipeline: validate upload -> normalize (optional) -> compute_speaker_embedding -> store.
"""
import logging
import tempfile
from pathlib import Path
from typing import Optional

from config import CLONE_MAX_DURATION_SEC, CLONE_MIN_DURATION_SEC
from kani_tts import compute_speaker_embedding
from voice_store import create_voice_id, save_embedding


def _get_duration_sec(audio_path: str) -> float:
    try:
        import torchaudio
        wav, sr = torchaudio.load(audio_path)
        return wav.shape[-1] / float(sr)
    except Exception as e:
        raise ValueError(f"Could not load audio: {e!s}") from e


def clone_voice(
    audio_path: str,
    consent_scope: str = "tts",
    name: Optional[str] = None,
    owner_id: Optional[str] = None,
) -> str:
    """
    Validate audio, compute speaker embedding, store and return voice_id.

    Args:
        audio_path: Path to WAV/MP3 (or any format torchaudio loads).
        consent_scope: Stored in metadata (e.g. "tts", "commercial").
        name: Optional display name for the voice.
        owner_id: Optional owner for per-user scoping when DB is used.

    Returns:
        voice_id (UUID string).

    Raises:
        ValueError: If duration out of range or audio invalid.
    """
    duration = _get_duration_sec(audio_path)
    if duration < CLONE_MIN_DURATION_SEC:
        raise ValueError(f"Audio too short: {duration:.1f}s (min {CLONE_MIN_DURATION_SEC}s)")
    if duration > CLONE_MAX_DURATION_SEC:
        raise ValueError(f"Audio too long: {duration:.1f}s (max {CLONE_MAX_DURATION_SEC}s)")

    try:
        embedding = compute_speaker_embedding(audio_path)
    except Exception as e:
        logging.exception("Speaker embedding failed")
        raise RuntimeError(f"Voice extraction failed: {e!s}") from e

    voice_id = create_voice_id()
    save_embedding(voice_id, embedding, consent_scope=consent_scope, name=name, owner_id=owner_id)
    return voice_id
