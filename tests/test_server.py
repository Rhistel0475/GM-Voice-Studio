"""Minimal tests for Kani TTS API endpoint behavior without HTTP client integration."""
import pytest
from fastapi import HTTPException

from server import health, ready


def test_health_payload():
    """health() returns status and service keys used by monitoring checks."""
    data = health()
    assert data.get("status") == "ok"
    assert data.get("service") == "kani-tts"


def test_ready_raises_503_when_model_not_loaded(monkeypatch):
    """ready() raises HTTP 503 if TTS model is not loaded yet."""
    import tts_service

    monkeypatch.setattr(tts_service, "is_model_loaded", lambda: False)
    with pytest.raises(HTTPException) as excinfo:
        ready()
    assert excinfo.value.status_code == 503
    assert "not yet loaded" in str(excinfo.value.detail).lower()


def test_ready_returns_ok_when_model_loaded(monkeypatch):
    """ready() returns ready status once model is loaded."""
    import tts_service

    monkeypatch.setattr(tts_service, "is_model_loaded", lambda: True)
    data = ready()
    assert data == {"status": "ready"}
