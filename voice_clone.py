"""
Voice clone pipeline: validate upload -> Pocket TTS voice state -> export to .safetensors -> store.
"""
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

from config import CLONE_MAX_DURATION_SEC, CLONE_MIN_DURATION_SEC
from voice_store import create_voice_id, save_voice_from_file
from tts_service import _get_tts


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
    faction: Optional[str] = None,
) -> str:
    """
    Validate audio, compute Pocket TTS voice state, export to .safetensors, store and return voice_id.
    """
    duration = _get_duration_sec(audio_path)
    if duration < CLONE_MIN_DURATION_SEC:
        raise ValueError(f"Audio too short: {duration:.1f}s (min {CLONE_MIN_DURATION_SEC}s)")
    if duration > CLONE_MAX_DURATION_SEC:
        raise ValueError(f"Audio too long: {duration:.1f}s (max {CLONE_MAX_DURATION_SEC}s)")

    voice_id = create_voice_id()
    tmp_path = None
    try:
        model = _get_tts()
        voice_state = model.get_state_for_audio_prompt(audio_path)
        from pocket_tts import export_model_state
        fd, tmp_path = tempfile.mkstemp(suffix=".safetensors")
        os.close(fd)
        export_model_state(voice_state, tmp_path)
        save_voice_from_file(
            voice_id,
            tmp_path,
            consent_scope=consent_scope,
            name=name,
            owner_id=owner_id,
            faction=faction,
        )
    except Exception as e:
        logging.exception("Voice extraction failed")
        raise RuntimeError(f"Voice extraction failed: {e!s}") from e
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return voice_id
