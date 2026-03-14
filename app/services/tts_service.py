"""
Thin TTS dispatcher over provider-specific implementations.
"""
from __future__ import annotations

import os
import tempfile
from typing import Optional

import soundfile as sf

from app.core.config import AUDIO_CACHE_SIZE, TTS_PROVIDER
from app.services import tts_hume, tts_kani

DEFAULT_TTS_TEMPERATURE = tts_kani.DEFAULT_TTS_TEMPERATURE
DEFAULT_TTS_TOP_P = tts_kani.DEFAULT_TTS_TOP_P
DEFAULT_TTS_REPETITION_PENALTY = tts_kani.DEFAULT_TTS_REPETITION_PENALTY
DEFAULT_LANGUAGE_TAG = tts_kani.DEFAULT_LANGUAGE_TAG

_audio_cache: list[str] = []


def get_tts_provider() -> str:
    return TTS_PROVIDER


def is_hume_provider() -> bool:
    return get_tts_provider() == "hume"


def _provider_module():
    return tts_hume if is_hume_provider() else tts_kani


def _voice_capabilities(*, deletable: bool, editable: bool, clonable: bool) -> dict[str, bool]:
    return {
        "narrate": True,
        "preview": True,
        "assign": True,
        "delete": deletable,
        "edit": editable,
        "clone": clonable,
    }


def normalize_stored_voice(voice: dict) -> dict:
    item = dict(voice)
    item.setdefault("source", "cloned")
    item["provider"] = "kani"
    item["provider_kind"] = "local_embedding"
    item["capabilities"] = _voice_capabilities(deletable=True, editable=True, clonable=True)
    item["deletable"] = True
    item["editable"] = True
    return item


def make_hume_voice_id(provider_kind: str, voice_id: str) -> str:
    return tts_hume.make_hume_voice_id(provider_kind, voice_id)


def parse_hume_voice_id(voice_id: Optional[str]) -> Optional[tuple[str, str]]:
    return tts_hume.parse_hume_voice_id(voice_id)


def resolve_hume_voice_id(voice_id: Optional[str]) -> Optional[str]:
    return tts_hume.resolve_hume_voice_id(voice_id)


def list_hume_voices() -> list[dict]:
    return tts_hume.list_hume_voices()


def get_hume_voice(voice_id: str) -> Optional[dict]:
    return tts_hume.get_hume_voice(voice_id)


def delete_hume_voice(voice_id: str) -> bool:
    return tts_hume.delete_hume_voice(voice_id)


def is_model_loaded() -> bool:
    return _provider_module().is_model_loaded()


def get_supported_language_tags() -> list[str]:
    return _provider_module().get_supported_language_tags()


def get_preset_voices() -> list[str]:
    return _provider_module().get_preset_voices()


def _is_preset_voice(voice_id: str) -> bool:
    return _provider_module().is_preset_voice(voice_id)


def generate(
    text: str,
    language_tag: Optional[str] = DEFAULT_LANGUAGE_TAG,
    speaker_emb_path: Optional[str] = None,
    temperature: float = DEFAULT_TTS_TEMPERATURE,
    top_p: float = DEFAULT_TTS_TOP_P,
    repetition_penalty: float = DEFAULT_TTS_REPETITION_PENALTY,
):
    return _provider_module().generate(
        text,
        language_tag=language_tag,
        speaker_emb_path=speaker_emb_path,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )


def _evict_old_audio():
    while len(_audio_cache) >= AUDIO_CACHE_SIZE and _audio_cache:
        path = _audio_cache.pop(0)
        try:
            os.unlink(path)
        except OSError:
            pass


def generate_to_file(
    text: str,
    language_tag: Optional[str] = DEFAULT_LANGUAGE_TAG,
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
