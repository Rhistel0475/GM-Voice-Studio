"""
Hume TTS provider implementation.
"""
from __future__ import annotations

import io
import json
import logging
import urllib.error
import urllib.request
from typing import Optional

import numpy as np
import soundfile as sf

from app.core.config import DEFAULT_VOICE_ID, HUME_API_KEY, HUME_BASE_URL, HUME_TTS_VERSION

HUME_SAMPLE_RATE = 24000
HUME_VOICE_ID_PREFIX = "hume"
HUME_PROVIDER_HUME_AI = "HUME_AI"
HUME_PROVIDER_CUSTOM_VOICE = "CUSTOM_VOICE"


def make_hume_voice_id(provider_kind: str, voice_id: str) -> str:
    return f"{HUME_VOICE_ID_PREFIX}:{provider_kind}:{voice_id}"


def parse_hume_voice_id(voice_id: Optional[str]) -> Optional[tuple[str, str]]:
    raw = (voice_id or "").strip()
    if not raw.startswith(f"{HUME_VOICE_ID_PREFIX}:"):
        return None
    parts = raw.split(":", 2)
    if len(parts) != 3 or not parts[1] or not parts[2]:
        return None
    return parts[1], parts[2]


def is_hume_voice_id(voice_id: Optional[str]) -> bool:
    return parse_hume_voice_id(voice_id) is not None


def _voice_capabilities(*, deletable: bool, editable: bool) -> dict[str, bool]:
    return {
        "narrate": True,
        "preview": True,
        "assign": True,
        "delete": deletable,
        "edit": editable,
        "clone": False,
    }


def _build_hume_voice(
    *,
    remote_id: str,
    provider_kind: str,
    name: str,
    description: str,
    tags: list[str],
    featured: bool,
) -> dict:
    deletable = provider_kind == HUME_PROVIDER_CUSTOM_VOICE and not featured
    editable = provider_kind == HUME_PROVIDER_CUSTOM_VOICE and not featured
    return {
        "voice_id": make_hume_voice_id(provider_kind, remote_id),
        "provider_voice_id": remote_id,
        "provider": "hume",
        "provider_kind": provider_kind,
        "name": name,
        "source": "system" if provider_kind == HUME_PROVIDER_HUME_AI else "custom",
        "status": "ready",
        "description": description,
        "tags": tags,
        "icon": "crown" if featured else None,
        "featured": featured,
        "capabilities": _voice_capabilities(deletable=deletable, editable=editable),
        "deletable": deletable,
        "editable": editable,
    }


def _hume_headers() -> dict[str, str]:
    if not HUME_API_KEY:
        raise RuntimeError("Hume API key not configured. Set HUME_API_KEY.")
    return {
        "X-Hume-Api-Key": HUME_API_KEY,
        "Content-Type": "application/json",
        "Accept": "*/*",
        "User-Agent": "GM-Voice-Studio/1.0 (+https://api.hume.ai)",
    }


def _hume_request(path: str, payload: Optional[dict] = None, method: str = "GET") -> bytes:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        f"{HUME_BASE_URL.rstrip('/')}{path}",
        data=data,
        method=method,
        headers=_hume_headers(),
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Hume API request failed ({exc.code}): {detail or exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach Hume API: {exc.reason}") from exc


def list_hume_voices() -> list[dict]:
    if not HUME_API_KEY:
        logging.warning("Hume voice list requested but HUME_API_KEY is not configured.")
        return []

    voices: list[dict] = []
    try:
        for provider_kind in (HUME_PROVIDER_HUME_AI, HUME_PROVIDER_CUSTOM_VOICE):
            raw = _hume_request(f"/v0/tts/voices?provider={provider_kind}")
            payload = json.loads(raw.decode("utf-8"))
            items = payload.get("voices_page") or payload.get("voices") or payload.get("items") or []
            logging.info("Fetched %d Hume voices for provider=%s", len(items), provider_kind)
            for item in items:
                remote_id = str(item.get("id") or item.get("voice_id") or "").strip()
                if not remote_id:
                    continue
                featured = bool(DEFAULT_VOICE_ID and remote_id == DEFAULT_VOICE_ID.strip())
                name = item.get("name") or item.get("display_name") or item.get("description") or remote_id
                description = item.get("description") or ""
                tags: list[str] = []
                if featured:
                    name = "Master Voice"
                    description = description or "Featured Hume voice for primary narration and character performance."
                    tags = ["featured", "master-voice"]
                voices.append(
                    _build_hume_voice(
                        remote_id=remote_id,
                        provider_kind=provider_kind,
                        name=str(name).strip() or remote_id,
                        description=description,
                        tags=tags,
                        featured=featured,
                    )
                )
    except Exception as exc:
        logging.warning("Failed to fetch Hume voices: %s", exc)

    preferred_id = None
    if DEFAULT_VOICE_ID:
        raw_preferred = DEFAULT_VOICE_ID.strip()
        for voice in voices:
            if raw_preferred in {
                (voice.get("voice_id") or "").strip(),
                (voice.get("provider_voice_id") or "").strip(),
            }:
                preferred_id = voice.get("voice_id")
                break
        if preferred_id is None:
            synthetic_voice = _build_hume_voice(
                remote_id=raw_preferred,
                provider_kind=HUME_PROVIDER_HUME_AI,
                name="Master Voice",
                description="Featured Hume voice configured locally. If playback fails, verify this voice ID exists in your Hume account or Voice Library access.",
                tags=["featured", "master-voice"],
                featured=True,
            )
            voices.append(synthetic_voice)
            preferred_id = synthetic_voice["voice_id"]

    if preferred_id:
        voices.sort(key=lambda voice: 0 if voice.get("voice_id") == preferred_id else 1)
    logging.info("Resolved %d total Hume voices", len(voices))
    return voices


def resolve_hume_voice_id(voice_id: Optional[str]) -> Optional[str]:
    raw = (voice_id or "").strip()
    if not raw:
        return None
    if is_hume_voice_id(raw):
        return raw

    matches = [voice for voice in list_hume_voices() if (voice.get("provider_voice_id") or "").strip() == raw]
    if len(matches) == 1:
        return matches[0]["voice_id"]
    return None


def get_hume_voice(voice_id: str) -> Optional[dict]:
    resolved = resolve_hume_voice_id(voice_id)
    wanted = parse_hume_voice_id(resolved)
    if not wanted:
        return None
    for voice in list_hume_voices():
        current = parse_hume_voice_id(voice.get("voice_id"))
        if current == wanted:
            return voice
    return None


def delete_hume_voice(voice_id: str) -> bool:
    parsed = parse_hume_voice_id(resolve_hume_voice_id(voice_id))
    if not parsed:
        return False
    provider_kind, remote_id = parsed
    if provider_kind != HUME_PROVIDER_CUSTOM_VOICE:
        raise ValueError("Built-in Hume voices cannot be deleted.")
    _hume_request(f"/v0/tts/voices/{remote_id}", method="DELETE")
    return True


def is_model_loaded() -> bool:
    return bool(HUME_API_KEY)


def get_supported_language_tags() -> list[str]:
    return ["en"]


def get_preset_voices() -> list[str]:
    return [voice["voice_id"] for voice in list_hume_voices()]


def is_preset_voice(voice_id: str) -> bool:
    return resolve_hume_voice_id(voice_id) is not None


def generate(
    text: str,
    language_tag: Optional[str] = "en",
    speaker_emb_path: Optional[str] = None,
    temperature: float = 0.65,
    top_p: float = 0.80,
    repetition_penalty: float = 1.15,
) -> tuple[np.ndarray, int]:
    del language_tag, temperature, top_p, repetition_penalty
    text = (text or "").strip()
    if not text:
        raise ValueError("Text is required")

    voice_payload = None
    resolved_voice_id = resolve_hume_voice_id(speaker_emb_path)
    if not resolved_voice_id and DEFAULT_VOICE_ID:
        resolved_voice_id = resolve_hume_voice_id(DEFAULT_VOICE_ID)

    parsed = parse_hume_voice_id(resolved_voice_id)
    if parsed:
        provider_kind, remote_id = parsed
        voice_payload = {"id": remote_id}
        logging.info("Generating Hume TTS with voice_id=%s provider=%s", remote_id, provider_kind)
    else:
        logging.info("Generating Hume TTS without explicit voice_id")

    payload: dict = {
        "utterances": [{"text": text}],
        "format": {"type": "wav"},
        "version": HUME_TTS_VERSION,
    }
    if voice_payload is not None:
        payload["utterances"][0]["voice"] = voice_payload
    elif HUME_TTS_VERSION == "2":
        raise ValueError("Hume Octave 2 requires a voice. Check DEFAULT_VOICE_ID or select a voice from /voices/list.")

    audio_bytes = _hume_request("/v0/tts/file", payload=payload, method="POST")
    logging.info("Received %d bytes from Hume TTS", len(audio_bytes))
    audio, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32), int(sample_rate or HUME_SAMPLE_RATE)
