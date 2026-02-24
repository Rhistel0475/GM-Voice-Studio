"""
TTS service: thin interface over the XTTSv2 engine.
Callers get (audio_array, sample_rate); engine can be swapped later.
"""
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import soundfile as sf

from config import (
    AUDIO_CACHE_SIZE,
    CLONE_GPT_COND_LEN,
    CLONE_MAX_REF_LENGTH,
    CLONE_SOUND_NORM_REFS,
    CLONE_TRIM_DB,
)

# XTTSv2 Supported Languages
DEFAULT_LANGUAGE_TAGS = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi"]

_tts = None
_audio_cache: list[str] = []

def _get_tts():
    global _tts
    if _tts is None:
        os.environ["COQUI_TOS_AGREED"] = "1"
        from TTS.api import TTS
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Loading XTTSv2 on {device}...")
        _tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    return _tts

def is_model_loaded() -> bool:
    return _tts is not None

def get_supported_language_tags() -> list[str]:
    return list(DEFAULT_LANGUAGE_TAGS)


def get_conditioning_latents_for_clone(audio_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute XTTSv2 conditioning latents from reference audio using clone-quality config."""
    tts = _get_tts()
    gpt_cond_chunk_len = min(6, CLONE_GPT_COND_LEN)
    return tts.synthesizer.tts_model.get_conditioning_latents(
        audio_path=audio_path,
        max_ref_length=CLONE_MAX_REF_LENGTH,
        gpt_cond_len=CLONE_GPT_COND_LEN,
        gpt_cond_chunk_len=gpt_cond_chunk_len,
        sound_norm_refs=CLONE_SOUND_NORM_REFS,
        librosa_trim_db=CLONE_TRIM_DB,
    )

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
    temperature: float = 0.65,       # Lowered from 0.75 for stability
    top_p: float = 0.80,             # Lowered from 0.85
    repetition_penalty: float = 1.15, # Lowered from 2.0 to stop slurring
) -> tuple[np.ndarray, int]:
    
    text = (text or "").strip()
    if not text:
        raise ValueError("Text is required")
    if not speaker_emb_path or not Path(speaker_emb_path).exists():
        raise ValueError("XTTSv2 requires a Character/NPC voice to be selected to generate audio.")

    tts = _get_tts()
    lang = language_tag if language_tag in DEFAULT_LANGUAGE_TAGS else "en"

    try:
        latents = torch.load(speaker_emb_path)
        if not isinstance(latents, dict):
            raise ValueError("Voice file is in an old format. Re-clone the voice by uploading a new sample.")
            
        device = tts.synthesizer.tts_model.device
        
        # Send latents to GPU/CPU
        gpt_cond_latent = latents["gpt_cond_latent"].to(device)
        speaker_embedding = latents["speaker_embedding"].to(device)

        # Strict keyword arguments to prevent Tensor/Boolean evaluation bugs
        out = tts.synthesizer.tts_model.inference(
            text=text,
            language=lang,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )
        audio = np.array(out["wav"])
        sr = 24000  # XTTSv2 native sample rate
    except Exception as e:
        logging.exception("TTS generate failed")
        raise RuntimeError(f"Generation failed: {e!s}") from e

    return audio, sr

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
        try: os.unlink(path)
        except OSError: pass
        raise RuntimeError(f"Could not save audio: {e!s}") from e
    _audio_cache.append(path)
    return path