"""
Transcription adapter: abstract interface and Deepgram implementation.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Protocol, runtime_checkable


@runtime_checkable
class TranscriptionAdapter(Protocol):
    """Interface for speech-to-text (e.g. Deepgram)."""

    def transcribe(self, audio_bytes: bytes, mime_type: str = "audio/webm") -> str:
        """Return transcript text for the given audio. Empty if no speech."""
        ...


class DeepgramTranscriptionAdapter:
    """Deepgram REST API implementation of TranscriptionAdapter."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        language: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._language = language

    def transcribe(self, audio_bytes: bytes, mime_type: str = "audio/webm") -> str:
        from app.core.config import DEEPGRAM_API_KEY, DEEPGRAM_MODEL, DEEPGRAM_LANGUAGE
        key = self._api_key if self._api_key is not None else DEEPGRAM_API_KEY
        if not key:
            raise RuntimeError("Deepgram is not configured. Set DEEPGRAM_API_KEY in your environment.")
        if not audio_bytes:
            return ""

        model = self._model if self._model is not None else (DEEPGRAM_MODEL or "nova-3")
        params = {
            "model": model,
            "punctuate": "true",
            "smart_format": "true",
        }
        lang = self._language if self._language is not None else DEEPGRAM_LANGUAGE
        if lang:
            params["language"] = lang

        query = urllib.parse.urlencode(params)
        url = f"https://api.deepgram.com/v1/listen?{query}"
        request = urllib.request.Request(url, data=audio_bytes, method="POST")
        request.add_header("Authorization", f"Token {key}")
        request.add_header("Content-Type", (mime_type or "audio/webm").strip())

        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Deepgram request failed ({exc.code}): {detail or exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Deepgram connection error: {exc.reason}") from exc

        results = payload.get("results") or {}
        channels = results.get("channels") or []
        if not channels:
            return ""
        alternatives = channels[0].get("alternatives") or []
        if not alternatives:
            return ""
        return (alternatives[0].get("transcript") or "").strip()


_default_transcription_adapter: DeepgramTranscriptionAdapter | None = None


def get_default_transcription_adapter() -> TranscriptionAdapter:
    """Return the default transcription adapter (Deepgram)."""
    global _default_transcription_adapter
    if _default_transcription_adapter is None:
        _default_transcription_adapter = DeepgramTranscriptionAdapter()
    return _default_transcription_adapter
