"""
Celery app for async voice clone. Optional: set CELERY_BROKER_URL to enable.
Run worker: celery -A celery_app worker -l info
"""
import os

from celery import Celery

from config import CELERY_BROKER_URL

app = Celery(
    "kani_tts",
    broker=CELERY_BROKER_URL or "memory://",
    backend=CELERY_BROKER_URL or "cache+memory://",
)
app.conf.task_default_queue = "kani_tts"
app.conf.result_expires = 86400  # 24h


@app.task(bind=True)
def clone_voice_task(
    self,
    upload_path: str,
    consent_scope: str = "tts",
    name: str | None = None,
    owner_id: str | None = None,
):
    """
    Run voice clone on a pre-saved upload file. Returns voice_id.
    Caller must save the upload to upload_path before enqueuing.
    """
    from voice_clone import clone_voice

    try:
        voice_id = clone_voice(
            upload_path,
            consent_scope=consent_scope,
            name=name,
            owner_id=owner_id,
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
    language_tag: str = "en_us",
    voice_id: str | None = None,
    chunk_by: str = "sentence",
    max_chars: int = 500,
):
    """
    Run long-form narrate: split text, TTS each chunk, concatenate, write WAV to NARRATE_RESULT_PATH/job_id.wav.
    Returns {"job_type": "narrate"} on success.
    """
    from config import NARRATE_RESULT_PATH
    from text_utils import MAX_CHUNKS, MAX_TOTAL_CHARS, split_for_tts
    from tts_service import generate as tts_generate
    from voice_store import load_embedding_path

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

        speaker_emb_path = load_embedding_path(voice_id) if voice_id else None
        if voice_id and not speaker_emb_path:
            return {"job_type": "narrate", "status": "failed", "error": "Voice not found"}

        import numpy as np
        import soundfile as sf

        audio_list = []
        sr_out = None
        for chunk in chunks:
            audio, sr = tts_generate(
                chunk,
                language_tag=language_tag,
                speaker_emb_path=speaker_emb_path,
                temperature=0.95,
                top_p=0.9,
                repetition_penalty=1.15,
            )
            if sr_out is None:
                sr_out = sr
            audio_list.append(audio)

        concatenated = np.concatenate(audio_list)
        sf.write(out_path, concatenated, sr_out, format="WAV")
        return {"job_type": "narrate", "status": "completed"}
    except Exception as e:
        return {"job_type": "narrate", "status": "failed", "error": str(e)}
