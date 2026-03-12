"""
Voice clone pipeline: validate upload -> KaniTTS-2 speaker embedding -> store as .pt.
"""
import logging
from typing import Optional

from app.core.config import CLONE_MAX_DURATION_SEC, CLONE_MIN_DURATION_SEC
from app.services.voice_store_service import create_voice_id, save_embedding


def _get_duration_sec(audio_path: str) -> float:
    try:
        import torchaudio
        wav, sr = torchaudio.load(audio_path)
        return wav.shape[-1] / float(sr)
    except Exception as e:
        raise ValueError(f"Could not load audio: {e!s}") from e


def _apply_speaker_embedder_compat():
    """Ensure kani_tts EmbeddingsModel is compatible with current transformers (all_tied_weights_keys)."""
    from kani_tts import speaker_embedder
    EmbeddingsModel = speaker_embedder.EmbeddingsModel
    if getattr(EmbeddingsModel, "_all_tied_weights_keys_patched", False):
        return
    _orig_init = EmbeddingsModel.__init__

    def _patched_init(self, config):
        _orig_init(self, config)
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}

    EmbeddingsModel.__init__ = _patched_init
    EmbeddingsModel._all_tied_weights_keys_patched = True


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
        _apply_speaker_embedder_compat()
        from kani_tts import SpeakerEmbedder
        embedder = SpeakerEmbedder(
            model_name="nineninesix/speaker-emb-tbr",
            max_duration_sec=CLONE_MAX_DURATION_SEC,
        )
        speaker_embedding = embedder.embed_audio_file(audio_path)
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
