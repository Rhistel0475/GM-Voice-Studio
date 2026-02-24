"""
Voice clone pipeline: validate upload -> compute_speaker_embedding -> store.
"""
import logging
import tempfile
from pathlib import Path
from typing import Optional

from config import CLONE_MAX_DURATION_SEC, CLONE_MIN_DURATION_SEC
from voice_store import create_voice_id, save_embedding
from tts_service import _get_tts  # Import our new XTTSv2 service


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
    faction: Optional[str] = None,  # <-- ADDED FACTION PARAMETER HERE
) -> str:
    """
    Validate audio, compute XTTS latents, store and return voice_id.
    """
    duration = _get_duration_sec(audio_path)
    if duration < CLONE_MIN_DURATION_SEC:
        raise ValueError(f"Audio too short: {duration:.1f}s (min {CLONE_MIN_DURATION_SEC}s)")
    if duration > CLONE_MAX_DURATION_SEC:
        raise ValueError(f"Audio too long: {duration:.1f}s (max {CLONE_MAX_DURATION_SEC}s)")

    try:
        # Load the XTTSv2 model
        tts = _get_tts()
        
        # XTTSv2 computes two separate latent tensors for voice cloning
        gpt_cond_latent, speaker_embedding = tts.synthesizer.tts_model.get_conditioning_latents(audio_path=audio_path)
        
        # Store them together in a dictionary
        embedding = {
            "gpt_cond_latent": gpt_cond_latent,
            "speaker_embedding": speaker_embedding
        }
    except Exception as e:
        logging.exception("Speaker embedding failed")
        raise RuntimeError(f"Voice extraction failed: {e!s}") from e

    voice_id = create_voice_id()
    
    # Pass faction down to your database/storage
    save_embedding(
        voice_id, 
        embedding, 
        consent_scope=consent_scope, 
        name=name, 
        owner_id=owner_id, 
        faction=faction  # <-- PASSED FACTION HERE
    )
    return voice_id
