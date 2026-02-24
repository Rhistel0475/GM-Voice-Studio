"""
TTS service: thin interface over XTTSv2 (Coqui TTS).
Callers get (audio_array, sample_rate). Set COQUI_TOS_AGREED=1 to accept the model license.
"""
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import soundfile as sf

# Coqui TTS imports isin_mps_friendly from transformers.pytorch_utils, which was removed in transformers 5.x.
# Provide a compatibility shim so TTS can load (torch.isin works on MPS in recent PyTorch).
import transformers.pytorch_utils as _tf_pt_utils
if not hasattr(_tf_pt_utils, "isin_mps_friendly"):
    _tf_pt_utils.isin_mps_friendly = torch.isin

from config import AUDIO_CACHE_SIZE

# XTTSv2 supported languages (no preset accents; always use a cloned voice)
DEFAULT_LANGUAGE_TAGS = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi"]

_tts = None
_audio_cache: list[str] = []


def _get_tts():
    global _tts
    if _tts is None:
        os.environ["COQUI_TOS_AGREED"] = "1"
        from TTS.api import TTS
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info("Loading XTTSv2 on %s...", device)
        _tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    return _tts


def is_model_loaded() -> bool:
    return _tts is not None


def get_supported_language_tags() -> list[str]:
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
    language_tag: Optional[str] = "en",
    speaker_emb_path: Optional[str] = None,
    temperature: float = 0.75,
    top_p: float = 0.85,
    repetition_penalty: float = 2.0,
) -> tuple[np.ndarray, int]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Text is required")
    if not speaker_emb_path or not Path(speaker_emb_path).exists():
        raise ValueError("XTTSv2 requires a Character/NPC voice to be selected to generate audio.")

    tts = _get_tts()
    lang = language_tag if language_tag in DEFAULT_LANGUAGE_TAGS else "en"

    try:
        try:
            latents = torch.load(speaker_emb_path, map_location="cpu", weights_only=False)
        except TypeError:
            latents = torch.load(speaker_emb_path, map_location="cpu")
        device = tts.synthesizer.tts_model.device
        # Support dict (our format + TTS internal "gpt_conditioning_latents") or tuple from get_conditioning_latents
        if isinstance(latents, dict):
            gpt_cond_latent = (latents.get("gpt_cond_latent") or latents.get("gpt_conditioning_latents"))
            speaker_embedding = latents.get("speaker_embedding")
            if gpt_cond_latent is None or speaker_embedding is None:
                raise ValueError(
                    "Voice file missing gpt_cond_latent or speaker_embedding. Re-clone the voice (upload a new sample)."
                )
        elif isinstance(latents, (tuple, list)) and len(latents) == 2:
            gpt_cond_latent, speaker_embedding = latents[0], latents[1]
        else:
            raise ValueError(
                "Voice file is in an old format (single embedding). Re-clone the voice by uploading a new sample."
            )
        gpt_cond_latent = gpt_cond_latent.to(device)
        speaker_embedding = speaker_embedding.to(device)

        out = tts.synthesizer.tts_model.inference(
            text,
            lang,
            gpt_cond_latent,
            speaker_embedding,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        audio = np.array(out["wav"])
        sr = 24000
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
        try:
            os.unlink(path)
        except OSError:
            pass
        raise RuntimeError(f"Could not save audio: {e!s}") from e
    _audio_cache.append(path)
    return path
