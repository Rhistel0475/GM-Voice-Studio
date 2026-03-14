"""
Typed configuration loaded from environment.
Single source of truth: get_settings() returns a cached Settings instance.
Module-level constants in config.py are kept for backward compatibility and are
populated from the same env; new code should use get_settings() or Depends(get_settings).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


def _root_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _str_list(env_key: str, default: str = "") -> list[str]:
    raw = os.environ.get(env_key, default).strip()
    return [k.strip() for k in raw.split(",") if k.strip()]


def _bool_env(env_key: str, default: bool = False) -> bool:
    return os.environ.get(env_key, "").strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class AppSettings:
    """Server bind and general app."""
    server_name: str = field(default_factory=lambda: os.environ.get("SERVER_NAME", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.environ.get("PORT", "7862")))


@dataclass(frozen=True)
class SecuritySettings:
    """API keys, admin key, abuse limits."""
    api_keys: list[str] = field(default_factory=lambda: _str_list("API_KEYS"))
    require_api_key: bool = field(default_factory=lambda: _bool_env("REQUIRE_API_KEY"))
    admin_api_key: str = field(default_factory=lambda: os.environ.get("ADMIN_API_KEY", "").strip())
    abuse_clone_per_ip_per_hour: int = field(
        default_factory=lambda: int(os.environ.get("ABUSE_CLONE_PER_IP_PER_HOUR", "0") or "0")
    )


@dataclass(frozen=True)
class DatabaseSettings:
    """Optional DB for voice metadata (SQLite or PostgreSQL URL)."""
    database_url: str = field(default_factory=lambda: os.environ.get("DATABASE_URL", "").strip())


@dataclass(frozen=True)
class StorageSettings:
    """Voice storage path and optional S3."""
    voice_storage_path: str = field(
        default_factory=lambda: os.environ.get("VOICE_STORAGE_PATH", str(_root_dir() / "voice_storage"))
    )
    voice_storage_backend: str = field(
        default_factory=lambda: (os.environ.get("VOICE_STORAGE_BACKEND", "local") or "local").lower()
    )
    voice_storage_bucket: str = field(default_factory=lambda: os.environ.get("VOICE_STORAGE_BUCKET", "").strip())
    aws_region: str = field(default_factory=lambda: os.environ.get("AWS_REGION", "us-east-1"))
    pending_clone_path: str = field(
        default_factory=lambda: os.environ.get("PENDING_CLONE_PATH", str(_root_dir() / "pending_clones"))
    )
    narrate_result_path: str = field(
        default_factory=lambda: os.environ.get("NARRATE_RESULT_PATH", str(_root_dir() / "narrate_results"))
    )


@dataclass(frozen=True)
class TTSSettings:
    """TTS and voice clone constraints."""
    provider: str = field(default_factory=lambda: (os.environ.get("TTS_PROVIDER", "kani") or "kani").strip().lower())
    default_voice_id: str = field(default_factory=lambda: os.environ.get("DEFAULT_VOICE_ID", "").strip())
    audio_cache_size: int = field(default_factory=lambda: int(os.environ.get("AUDIO_CACHE_SIZE", "10")))
    clone_min_duration_sec: float = field(default_factory=lambda: float(os.environ.get("CLONE_MIN_DURATION_SEC", "3.0")))
    clone_max_duration_sec: float = field(default_factory=lambda: float(os.environ.get("CLONE_MAX_DURATION_SEC", "60.0")))
    clone_target_sample_rate: int = 16000
    voice_retention_days: int = field(default_factory=lambda: int(os.environ.get("VOICE_RETENTION_DAYS", "0")))
    hume_api_key: str = field(default_factory=lambda: os.environ.get("HUME_API_KEY", "").strip())
    hume_secret_key: str = field(default_factory=lambda: os.environ.get("HUME_SECRET_KEY", "").strip())
    hume_base_url: str = field(default_factory=lambda: os.environ.get("HUME_BASE_URL", "https://api.hume.ai").strip())
    hume_version: str = field(default_factory=lambda: os.environ.get("HUME_TTS_VERSION", "2").strip())


@dataclass(frozen=True)
class LLMSettings:
    """Anthropic/Claude and adventure parsing."""
    anthropic_api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", "").strip())
    ai_model: str = field(default_factory=lambda: os.environ.get("AI_MODEL", "claude-haiku-4-5-20251001").strip())
    max_adventure_chars: int = field(default_factory=lambda: int(os.environ.get("MAX_ADVENTURE_CHARS", "160000")))


@dataclass(frozen=True)
class RetrievalSettings:
    """Pinecone and OpenAI embeddings for RAG."""
    openai_api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", "").strip())
    pinecone_api_key: str = field(default_factory=lambda: os.environ.get("PINECONE_API_KEY", "").strip())
    pinecone_index_name: str = field(default_factory=lambda: os.environ.get("PINECONE_INDEX_NAME", "co-dm-index").strip())


@dataclass(frozen=True)
class TranscriptionSettings:
    """Deepgram STT."""
    deepgram_api_key: str = field(default_factory=lambda: os.environ.get("DEEPGRAM_API_KEY", "").strip())
    deepgram_model: str = field(default_factory=lambda: os.environ.get("DEEPGRAM_MODEL", "nova-3").strip())
    deepgram_language: str = field(default_factory=lambda: os.environ.get("DEEPGRAM_LANGUAGE", "en-US").strip())
    auto_query_on_voice: bool = field(default_factory=lambda: _bool_env("AUTO_QUERY_ON_VOICE", True))


@dataclass(frozen=True)
class FrontendSettings:
    """CORS and rate limits."""
    cors_origins: str = field(default_factory=lambda: os.environ.get("CORS_ORIGINS", "").strip())
    rate_limit_global: str | None = field(
        default_factory=lambda: os.environ.get("RATE_LIMIT_GLOBAL", "60/minute").strip() or None
    )
    rate_limit_tts: str | None = field(
        default_factory=lambda: os.environ.get("RATE_LIMIT_TTS", "30/minute").strip() or None
    )
    rate_limit_clone: str | None = field(
        default_factory=lambda: os.environ.get("RATE_LIMIT_CLONE", "10/minute").strip() or None
    )


@dataclass(frozen=True)
class Settings:
    """Composed settings from environment."""
    app: AppSettings = field(default_factory=AppSettings)
    security: SecuritySettings = field(default_factory=SecuritySettings)
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    storage: StorageSettings = field(default_factory=StorageSettings)
    tts: TTSSettings = field(default_factory=TTSSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    transcription: TranscriptionSettings = field(default_factory=TranscriptionSettings)
    frontend: FrontendSettings = field(default_factory=FrontendSettings)
    # Legacy single-value (for backward compat and convenience)
    hf_token: str = field(default_factory=lambda: os.environ.get("HF_TOKEN", "").strip())
    celery_broker_url: str = field(default_factory=lambda: os.environ.get("CELERY_BROKER_URL", "").strip())


@lru_cache
def get_settings() -> Settings:
    """Return cached Settings instance. Load from env on first call."""
    return Settings()
