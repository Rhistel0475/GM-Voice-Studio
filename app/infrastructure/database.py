"""
SQLAlchemy database setup for GM Voice Studio campaign data.

- Local default: SQLite file `codm.db` in the project root.
- Production: set `DATABASE_URL` (e.g. Railway PostgreSQL). `postgres://` is normalized to `postgresql://`.

Schema is managed by Alembic (migrations/); do not create_all at runtime.
"""
import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Repo root: app/infrastructure/database.py -> parent.parent = app -> parent = root
_BASE_DIR = Path(__file__).resolve().parent.parent.parent

_env_db = (os.environ.get("DATABASE_URL") or "").strip()
if _env_db:
    # Heroku/Railway sometimes provide postgres://; SQLAlchemy 2 prefers postgresql://
    DATABASE_URL = _env_db.replace("postgres://", "postgresql://", 1)
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
else:
    DATABASE_URL = f"sqlite:///{_BASE_DIR / 'codm.db'}"
    # check_same_thread=False required for SQLite with FastAPI's thread-pool
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency that yields a DB session and closes it when done."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def run_alembic_upgrade() -> None:
    """Run Alembic migrations to head. Call from init_db so schema is up to date."""
    from alembic import command
    from alembic.config import Config

    alembic_cfg = Config(str(_BASE_DIR / "alembic.ini"))
    alembic_cfg.set_main_option("script_location", str(_BASE_DIR / "migrations"))
    alembic_cfg.set_main_option("sqlalchemy.url", DATABASE_URL)
    command.upgrade(alembic_cfg, "head")


def init_db() -> None:
    """Ensure campaign DB exists and schema is at latest migration (Alembic)."""
    from app.infrastructure.db_models import (  # noqa: F401 — registers models
        Campaign,
        GameSession,
        Location,
        NPC,
        Scene,
    )
    run_alembic_upgrade()
