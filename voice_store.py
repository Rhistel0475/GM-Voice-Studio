"""
Voice store: persist speaker embeddings (.pt) and metadata for voice_id.
Supports local directory (default) or optional S3 backend via VOICE_STORAGE_BACKEND=s3.
"""
import json
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

import torch

from config import (
    AWS_REGION,
    VOICE_STORAGE_BACKEND,
    VOICE_STORAGE_BUCKET,
    VOICE_STORAGE_PATH,
)

from db_voice import (
    use_db,
    db_insert_voice,
    db_get_voice,
    db_list_voices,
    db_update_voice,
    db_delete_voice,
)

if VOICE_STORAGE_BACKEND == "local":
    os.makedirs(VOICE_STORAGE_PATH, exist_ok=True)

# --- Local backend ---


def _pt_path(voice_id: str) -> Path:
    return Path(VOICE_STORAGE_PATH) / f"{voice_id}.pt"


def _meta_path(voice_id: str) -> Path:
    return Path(VOICE_STORAGE_PATH) / f"{voice_id}.json"


def _local_save_embedding(
    voice_id: str,
    embedding: dict | torch.Tensor,
    consent_scope: str = "tts",
    name: Optional[str] = None,
    faction: Optional[str] = None,
) -> None:
    pt = _pt_path(voice_id)
    meta = _meta_path(voice_id)
    if isinstance(embedding, dict):
        emb = {k: v.cpu() if hasattr(v, "cpu") else v for k, v in embedding.items()}
    else:
        emb = embedding.cpu()
        if emb.ndim == 1:
            emb = emb.unsqueeze(0)
    torch.save(emb, pt)
    meta.write_text(json.dumps({
        "voice_id": voice_id,
        "consent_scope": consent_scope,
        "created_at": time.time(),
        "name": (name or "").strip(),
        "faction": (faction or "").strip() or "",
    }, indent=2))


def _local_load_embedding_path(voice_id: str) -> Optional[str]:
    p = _pt_path(voice_id)
    if p.exists():
        return str(p)
    return None


def _local_get_metadata(voice_id: str) -> Optional[dict]:
    m = _meta_path(voice_id)
    if m.exists():
        return json.loads(m.read_text())
    return None


def _local_list_voices() -> list[dict]:
    out = []
    base = Path(VOICE_STORAGE_PATH)
    if not base.exists():
        return out
    for p in base.glob("*.json"):
        try:
            data = json.loads(p.read_text())
            out.append({
                "voice_id": data.get("voice_id", p.stem),
                "name": data.get("name", ""),
                "created_at": data.get("created_at", 0),
                "consent_scope": data.get("consent_scope", "tts"),
                "faction": data.get("faction", ""),
            })
        except (json.JSONDecodeError, OSError):
            continue
    out.sort(key=lambda x: x["created_at"], reverse=True)
    return out


def _local_update_metadata(voice_id: str, name: Optional[str] = None) -> bool:
    m = _meta_path(voice_id)
    if not m.exists():
        return False
    data = json.loads(m.read_text())
    if name is not None:
        data["name"] = (name or "").strip()
    m.write_text(json.dumps(data, indent=2))
    return True


def _local_delete_voice(voice_id: str) -> bool:
    pt = _pt_path(voice_id)
    meta = _meta_path(voice_id)
    ok = False
    if pt.exists():
        try:
            pt.unlink()
            ok = True
        except OSError as e:
            logging.warning("Could not delete %s: %s", pt, e)
    if meta.exists():
        try:
            meta.unlink()
            ok = True
        except OSError as e:
            logging.warning("Could not delete %s: %s", meta, e)
    return ok


# --- S3 backend ---

INDEX_KEY = "index.json"


def _s3_client():
    import boto3
    return boto3.client("s3", region_name=AWS_REGION)


def _s3_save_embedding(
    voice_id: str,
    embedding: dict | torch.Tensor,
    consent_scope: str = "tts",
    name: Optional[str] = None,
    faction: Optional[str] = None,
) -> None:
    client = _s3_client()
    bucket = VOICE_STORAGE_BUCKET
    if isinstance(embedding, dict):
        emb = {k: v.cpu() if hasattr(v, "cpu") else v for k, v in embedding.items()}
    else:
        emb = embedding.cpu()
        if emb.ndim == 1:
            emb = emb.unsqueeze(0)
    meta = {
        "voice_id": voice_id,
        "consent_scope": consent_scope,
        "created_at": time.time(),
        "name": (name or "").strip(),
        "faction": (faction or "").strip() or "",
    }
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(emb, f.name)
        try:
            with open(f.name, "rb") as g:
                client.put_object(Bucket=bucket, Key=f"{voice_id}.pt", Body=g.read())
        finally:
            try:
                os.unlink(f.name)
            except OSError:
                pass
    client.put_object(
        Bucket=bucket,
        Key=f"{voice_id}.json",
        Body=json.dumps(meta, indent=2),
        ContentType="application/json",
    )
    # Update index
    try:
        resp = client.get_object(Bucket=bucket, Key=INDEX_KEY)
        index = json.loads(resp["Body"].read().decode())
    except client.exceptions.NoSuchKey:
        index = []
    index_by_id = {e["voice_id"]: e for e in index}
    index_by_id[voice_id] = meta
    index = list(index_by_id.values())
    index.sort(key=lambda x: x["created_at"], reverse=True)
    client.put_object(
        Bucket=bucket,
        Key=INDEX_KEY,
        Body=json.dumps(index, indent=2),
        ContentType="application/json",
    )


def _s3_load_embedding_path(voice_id: str) -> Optional[str]:
    client = _s3_client()
    bucket = VOICE_STORAGE_BUCKET
    try:
        resp = client.get_object(Bucket=bucket, Key=f"{voice_id}.pt")
        data = resp["Body"].read()
    except client.exceptions.NoSuchKey:
        return None
    fd, path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    with open(path, "wb") as f:
        f.write(data)
    return path


def _s3_get_metadata(voice_id: str) -> Optional[dict]:
    client = _s3_client()
    bucket = VOICE_STORAGE_BUCKET
    try:
        resp = client.get_object(Bucket=bucket, Key=f"{voice_id}.json")
        return json.loads(resp["Body"].read().decode())
    except client.exceptions.NoSuchKey:
        return None


def _s3_list_voices() -> list[dict]:
    client = _s3_client()
    bucket = VOICE_STORAGE_BUCKET
    try:
        resp = client.get_object(Bucket=bucket, Key=INDEX_KEY)
        index = json.loads(resp["Body"].read().decode())
        return index
    except client.exceptions.NoSuchKey:
        return []


def _s3_update_metadata(voice_id: str, name: Optional[str] = None) -> bool:
    meta = _s3_get_metadata(voice_id)
    if not meta:
        return False
    if name is not None:
        meta["name"] = (name or "").strip()
    client = _s3_client()
    bucket = VOICE_STORAGE_BUCKET
    client.put_object(
        Bucket=bucket,
        Key=f"{voice_id}.json",
        Body=json.dumps(meta, indent=2),
        ContentType="application/json",
    )
    # Update index entry
    try:
        resp = client.get_object(Bucket=bucket, Key=INDEX_KEY)
        index = json.loads(resp["Body"].read().decode())
    except client.exceptions.NoSuchKey:
        return True
    for i, e in enumerate(index):
        if e.get("voice_id") == voice_id:
            index[i] = meta
            break
    client.put_object(
        Bucket=bucket,
        Key=INDEX_KEY,
        Body=json.dumps(index, indent=2),
        ContentType="application/json",
    )
    return True


def _s3_delete_voice(voice_id: str) -> bool:
    client = _s3_client()
    bucket = VOICE_STORAGE_BUCKET
    ok = False
    for key in (f"{voice_id}.pt", f"{voice_id}.json"):
        try:
            client.delete_object(Bucket=bucket, Key=key)
            ok = True
        except Exception as e:
            logging.warning("Could not delete s3 %s: %s", key, e)
    try:
        resp = client.get_object(Bucket=bucket, Key=INDEX_KEY)
        index = json.loads(resp["Body"].read().decode())
        index = [e for e in index if e.get("voice_id") != voice_id]
        client.put_object(
            Bucket=bucket,
            Key=INDEX_KEY,
            Body=json.dumps(index, indent=2),
            ContentType="application/json",
        )
    except client.exceptions.NoSuchKey:
        pass
    return ok


# --- Public API (dispatcher) ---

def _use_s3() -> bool:
    return VOICE_STORAGE_BACKEND == "s3" and bool(VOICE_STORAGE_BUCKET)


def create_voice_id() -> str:
    return str(uuid.uuid4())


def save_embedding(
    voice_id: str,
    embedding: torch.Tensor,
    consent_scope: str = "tts",
    name: Optional[str] = None,
    owner_id: Optional[str] = None,
    faction: Optional[str] = None,
) -> None:
    """Save .pt and metadata. Embedding shape [1, 128] or [128]. owner_id used when use_db() for per-user scoping."""
    created_at = time.time()
    if _use_s3():
        _s3_save_embedding(voice_id, embedding, consent_scope=consent_scope, name=name, faction=faction)
    else:
        _local_save_embedding(voice_id, embedding, consent_scope=consent_scope, name=name, faction=faction)
    if use_db():
        db_insert_voice(voice_id, (name or "").strip(), consent_scope, created_at, owner_id=owner_id, faction=faction)


def load_embedding_path(voice_id: str) -> Optional[str]:
    """Return path to .pt file if it exists. For S3, downloads to a temp file."""
    if _use_s3():
        return _s3_load_embedding_path(voice_id)
    return _local_load_embedding_path(voice_id)


def get_metadata(voice_id: str, owner_id: Optional[str] = None) -> Optional[dict]:
    """Return metadata dict if voice exists. When use_db() and owner_id set, only return if voice belongs to owner."""
    if use_db():
        return db_get_voice(voice_id, owner_id=owner_id)
    if _use_s3():
        return _s3_get_metadata(voice_id)
    return _local_get_metadata(voice_id)


def list_voices(owner_id: Optional[str] = None) -> list[dict]:
    """List voices: when use_db() and owner_id set, only voices owned by that owner; else all. Sorted by created_at desc."""
    if use_db():
        return db_list_voices(owner_id=owner_id)
    if _use_s3():
        return _s3_list_voices()
    return _local_list_voices()


def update_metadata(voice_id: str, name: Optional[str] = None, owner_id: Optional[str] = None) -> bool:
    """Update metadata (e.g. name). Returns False if voice not found or (when owner_id set) not owned by owner."""
    if use_db():
        return db_update_voice(voice_id, name=name, owner_id=owner_id)
    if _use_s3():
        return _s3_update_metadata(voice_id, name=name)
    return _local_update_metadata(voice_id, name=name)


def delete_voice(voice_id: str, owner_id: Optional[str] = None) -> bool:
    """Remove .pt and metadata for this voice. When use_db() and owner_id set, only delete if owned by owner. Returns True if something was deleted."""
    if use_db() and owner_id is not None:
        if not db_get_voice(voice_id, owner_id=owner_id):
            return False
    ok = _s3_delete_voice(voice_id) if _use_s3() else _local_delete_voice(voice_id)
    if use_db():
        ok = db_delete_voice(voice_id, owner_id=owner_id) or ok
    return ok
