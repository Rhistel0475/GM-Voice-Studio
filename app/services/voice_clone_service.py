"""
Voice clone pipeline: validate upload -> KaniTTS-2 speaker embedding -> store as .pt.
Uses soundfile for loading audio to avoid torchaudio/torchcodec FFmpeg dependency.
Loads kani_tts speaker_embedder module directly to avoid importing kani_tts.model
(which requires TransformersKwargs from newer transformers).
"""
import importlib.util
import logging
import os
from typing import Optional

import soundfile as sf

from app.core.config import CLONE_MAX_DURATION_SEC, CLONE_MIN_DURATION_SEC
from app.services.voice_store_service import create_voice_id, save_embedding

_speaker_embedder_module = None


def _get_duration_sec(audio_path: str) -> float:
    """Get duration in seconds using soundfile (no torchaudio/FFmpeg)."""
    try:
        info = sf.info(audio_path)
        return float(info.duration)
    except Exception as e:
        raise ValueError(f"Could not load audio: {e!s}") from e


def _load_audio_soundfile(audio_path: str):
    """Load (waveform, sample_rate) with soundfile; mono float in [-1, 1]."""
    data, sr = sf.read(audio_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, int(sr)


def _get_speaker_embedder_module():
    """
    Load kani_tts.speaker_embedder by file so we never run kani_tts/__init__.py
    (which pulls in model.py and TransformersKwargs). Apply compat to EmbeddingsModel.
    """
    global _speaker_embedder_module
    if _speaker_embedder_module is not None:
        return _speaker_embedder_module
    spec = importlib.util.find_spec("kani_tts")
    if spec is None or spec.origin is None:
        raise ImportError("kani_tts package not found")
    pkg_dir = os.path.dirname(spec.origin)
    spk_path = os.path.join(pkg_dir, "speaker_embedder.py")
    if not os.path.isfile(spk_path):
        raise ImportError(f"kani_tts speaker_embedder not found at {spk_path}")
    spk_spec = importlib.util.spec_from_file_location("kani_tts_speaker_embedder", spk_path)
    mod = importlib.util.module_from_spec(spk_spec)
    import sys
    sys.modules["kani_tts_speaker_embedder"] = mod
    spk_spec.loader.exec_module(mod)
    # Compat: EmbeddingsModel may need all_tied_weights_keys for current transformers
    EmbeddingsModel = getattr(mod, "EmbeddingsModel", None)
    if EmbeddingsModel and not getattr(EmbeddingsModel, "_all_tied_weights_keys_patched", False):
        _orig_init = EmbeddingsModel.__init__

        def _patched_init(self, config):
            _orig_init(self, config)
            if not hasattr(self, "all_tied_weights_keys"):
                self.all_tied_weights_keys = {}

        EmbeddingsModel.__init__ = _patched_init
        EmbeddingsModel._all_tied_weights_keys_patched = True
    _speaker_embedder_module = mod
    return mod


def clone_voice(
    audio_path: str,
    consent_scope: str = "tts",
    name: Optional[str] = None,
    owner_id: Optional[str] = None,
    faction: Optional[str] = None,
) -> str:
    """
    Validate audio, compute KaniTTS-2 speaker embedding, store as .pt and return voice_id.
    """
    duration = _get_duration_sec(audio_path)
    if duration < CLONE_MIN_DURATION_SEC:
        raise ValueError(f"Audio too short: {duration:.1f}s (min {CLONE_MIN_DURATION_SEC}s)")
    if duration > CLONE_MAX_DURATION_SEC:
        raise ValueError(f"Audio too long: {duration:.1f}s (max {CLONE_MAX_DURATION_SEC}s)")

    voice_id = create_voice_id()
    try:
        import torch
        speaker_embedder_mod = _get_speaker_embedder_module()
        SpeakerEmbedder = speaker_embedder_mod.SpeakerEmbedder
        embedder = SpeakerEmbedder(
            model_name="nineninesix/speaker-emb-tbr",
            max_duration_sec=CLONE_MAX_DURATION_SEC,
        )
        # Load with soundfile to avoid torchaudio/torchcodec FFmpeg dependency
        waveform, sample_rate = _load_audio_soundfile(audio_path)
        waveform_tensor = torch.from_numpy(waveform).float()
        speaker_embedding = embedder.embed_audio(waveform_tensor, sample_rate=sample_rate)
        save_embedding(
            voice_id,
            speaker_embedding,
            consent_scope=consent_scope,
            name=name,
            owner_id=owner_id,
            faction=faction,
        )
    except Exception as e:
        logging.exception("Voice extraction failed")
        raise RuntimeError(f"Voice extraction failed: {e!s}") from e

    return voice_id
