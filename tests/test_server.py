"""Minimal tests for Kani TTS API: health, readiness, optional TTS (slow)."""
import pytest
from fastapi.testclient import TestClient

from server import app

client = TestClient(app)


def test_health():
    """GET /health returns 200 and status ok."""
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "ok"
    assert data.get("service") == "kani-tts"


def test_ready_before_model_load():
    """GET /ready returns 503 until model has been loaded (or 200 if already loaded)."""
    r = client.get("/ready")
    # Before any TTS request, model is typically not loaded
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        assert r.json().get("status") == "ready"
    else:
        assert "not yet loaded" in r.text.lower() or r.status_code == 503


@pytest.mark.slow
def test_tts_accent_only():
    """POST /tts with text and accent returns 200 and WAV. Slow (loads model)."""
    r = client.post(
        "/tts",
        data={"text": "Hello.", "language_tag": "en_us"},
    )
    assert r.status_code == 200, r.text[:500]
    assert r.headers.get("content-type", "").startswith("audio/")
    assert len(r.content) > 1000  # non-trivial WAV
