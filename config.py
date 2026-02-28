"""Env-based configuration for TTS and voice storage."""
import os

# TTS engine (Pocket TTS via Kyutai)
AUDIO_CACHE_SIZE = int(os.environ.get("AUDIO_CACHE_SIZE", "10"))

# Server (Gradio and FastAPI)
SERVER_NAME = os.environ.get("SERVER_NAME", "0.0.0.0")
PORT = int(os.environ.get("PORT", "7862"))

# Voice cloning: where to store .safetensors voice files and metadata (local path for MVP)
VOICE_STORAGE_PATH = os.environ.get("VOICE_STORAGE_PATH", os.path.join(os.path.dirname(__file__), "voice_storage"))
# Optional object store: VOICE_STORAGE_BACKEND=local|s3 (default local)
VOICE_STORAGE_BACKEND = (os.environ.get("VOICE_STORAGE_BACKEND", "local") or "local").lower()
VOICE_STORAGE_BUCKET = os.environ.get("VOICE_STORAGE_BUCKET", "").strip()
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

# Optional DB for voice metadata (enables audit trail, future per-user voices). SQLite or PostgreSQL URL.
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()

# Optional queue for async voice clone. Set CELERY_BROKER_URL (e.g. redis://localhost:6379/0) to enable.
CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "").strip()
# Temp dir for uploads before worker processes (must be shared with worker if multi-host)
PENDING_CLONE_PATH = os.environ.get("PENDING_CLONE_PATH", os.path.join(os.path.dirname(__file__), "pending_clones"))
# Dir for async narrate WAV outputs (must be shared with worker if multi-host)
NARRATE_RESULT_PATH = os.environ.get("NARRATE_RESULT_PATH", os.path.join(os.path.dirname(__file__), "narrate_results"))

# Optional auth: comma-separated API keys (no key required if empty). Header: X-API-Key
API_KEYS = [k.strip() for k in os.environ.get("API_KEYS", "").split(",") if k.strip()]
# Require API key for non-public routes (health/voices list can stay public). Set 1 to require.
REQUIRE_API_KEY = os.environ.get("REQUIRE_API_KEY", "").strip() in ("1", "true", "yes")
# Admin key for take-down: X-Admin-Key. If set, DELETE /admin/voices/{voice_id} is allowed with this key.
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "").strip()
# Abuse: max clone requests per IP per hour (0 = disable)
ABUSE_CLONE_PER_IP_PER_HOUR = int(os.environ.get("ABUSE_CLONE_PER_IP_PER_HOUR", "0") or "0")

# Audio pipeline: clone sample constraints
CLONE_MIN_DURATION_SEC = float(os.environ.get("CLONE_MIN_DURATION_SEC", "3.0"))
CLONE_MAX_DURATION_SEC = float(os.environ.get("CLONE_MAX_DURATION_SEC", "120.0"))
CLONE_TARGET_SAMPLE_RATE = 16000

# GDPR: retention (document your policy; delete via DELETE /voices/{voice_id})
VOICE_RETENTION_DAYS = int(os.environ.get("VOICE_RETENTION_DAYS", "0"))  # 0 = keep until deleted

# Rate limits (e.g. "100/minute", "10/second"). Empty string = no limit.
RATE_LIMIT_GLOBAL = os.environ.get("RATE_LIMIT_GLOBAL", "60/minute") or None
RATE_LIMIT_TTS = os.environ.get("RATE_LIMIT_TTS", "30/minute") or None
RATE_LIMIT_CLONE = os.environ.get("RATE_LIMIT_CLONE", "10/minute") or None

# CORS: comma-separated origins (e.g. https://app.example.com). Empty = same-origin only.
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "").strip()

# Hugging Face token for gated models (e.g. Pocket TTS voice cloning). Set HF_TOKEN in env or .env.
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()

# Anthropic AI: Co-GM NPC dialogue generation
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
AI_MODEL = os.environ.get("AI_MODEL", "claude-opus-4-6").strip() or "claude-opus-4-6"
RATE_LIMIT_AI = os.environ.get("RATE_LIMIT_AI", "20/minute") or None
