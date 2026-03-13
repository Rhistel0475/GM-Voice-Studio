"""
Celery app for async voice clone. Optional: set CELERY_BROKER_URL to enable.
Run worker: celery -A app.infrastructure.tasks.celery_app worker -l info
"""
import os

from celery import Celery

from app.core.config import CELERY_BROKER_URL, NARRATE_RESULT_PATH

app = Celery(
    "pocket_tts",
    broker=CELERY_BROKER_URL or "memory://",
    backend=CELERY_BROKER_URL or "cache+memory://",
)
app.conf.task_default_queue = "pocket_tts"
app.conf.result_expires = 86400  # 24h


@app.task(bind=True)
def clone_voice_task(
    self,
    upload_path: str,
    consent_scope: str = "tts",
    name: str | None = None,
    owner_id: str | None = None,
    faction: str | None = None,
):
    """
    Run voice clone on a pre-saved upload file. Returns voice_id.
    Caller must save the upload to upload_path before enqueuing.
    """
    from app.services.voice_clone_service import clone_voice

    try:
        voice_id = clone_voice(
            upload_path,
            consent_scope=consent_scope,
            name=name,
            owner_id=owner_id,
            faction=faction,
        )
        return {"status": "completed", "voice_id": voice_id}
    except Exception as e:
        return {"status": "failed", "error": str(e)}
    finally:
        try:
            if os.path.exists(upload_path):
                os.unlink(upload_path)
        except OSError:
            pass


@app.task(bind=True)
def narrate_task(
    self,
    job_id: str,
    text: str,
    language_tag: str = "en",
    voice_id: str | None = None,
    chunk_by: str = "sentence",
    max_chars: int = 500,
):
    """
    Run long-form narrate: split text, TTS each chunk, concatenate, write WAV to NARRATE_RESULT_PATH/job_id.wav.
    voice_id can be a preset name or a cloned voice_id. Returns {"job_type": "narrate"} on success.
    """
    from app.core.text_utils import MAX_CHUNKS, MAX_TOTAL_CHARS, split_for_tts
    from app.services.tts_service import (
        DEFAULT_TTS_REPETITION_PENALTY,
        DEFAULT_TTS_TEMPERATURE,
        DEFAULT_TTS_TOP_P,
        _is_preset_voice,
        generate as tts_generate,
    )
    from app.services.voice_store_service import load_embedding_path

    if not voice_id:
        return {"job_type": "narrate", "status": "failed", "error": "Narrate requires voice_id."}
    if _is_preset_voice(voice_id):
        speaker_emb_path = voice_id.strip()
    else:
        speaker_emb_path = load_embedding_path(voice_id)
    if not speaker_emb_path:
        return {"job_type": "narrate", "status": "failed", "error": "Voice not found"}

    os.makedirs(NARRATE_RESULT_PATH, exist_ok=True)
    out_path = os.path.join(NARRATE_RESULT_PATH, f"{job_id}.wav")

    try:
        text = (text or "").strip()
        if not text or len(text) > MAX_TOTAL_CHARS:
            return {"job_type": "narrate", "status": "failed", "error": "Invalid or too long text"}
        chunks = split_for_tts(text, chunk_by=chunk_by, max_chars=max(50, min(max_chars, 1500)))
        if not chunks or len(chunks) > MAX_CHUNKS:
            chunks = chunks[:MAX_CHUNKS] if chunks else []
        if not chunks:
            return {"job_type": "narrate", "status": "failed", "error": "No chunks produced"}

        import numpy as np
        import soundfile as sf

        audio_list = []
        sr_out = None
        for chunk in chunks:
            audio, sr = tts_generate(
                chunk,
                language_tag=language_tag,
                speaker_emb_path=speaker_emb_path,
                temperature=DEFAULT_TTS_TEMPERATURE,
                top_p=DEFAULT_TTS_TOP_P,
                repetition_penalty=DEFAULT_TTS_REPETITION_PENALTY,
            )
            if sr_out is None:
                sr_out = sr
            audio_list.append(audio)

        concatenated = np.concatenate(audio_list)
        sf.write(out_path, concatenated, sr_out, format="WAV")
        return {"job_type": "narrate", "status": "completed"}
    except Exception as e:
        return {"job_type": "narrate", "status": "failed", "error": str(e)}
