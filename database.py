"""
SQLAlchemy database setup for GM Voice Studio campaign data.
Creates/opens codm.db (SQLite) in the project root.
"""
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

BASE_DIR = Path(__file__).resolve().parent
DATABASE_URL = f"sqlite:///{BASE_DIR / 'codm.db'}"

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


# Create all tables on first import (no-op if they already exist)
def init_db() -> None:
    from models import Campaign, NPC, Scene, Location  # noqa: F401 — registers models
    Base.metadata.create_all(bind=engine)
    _ensure_runtime_migrations()


def _ensure_runtime_migrations() -> None:
    """Apply lightweight SQLite migrations for columns added after initial deployment."""
    if engine.url.get_backend_name() != "sqlite":
        return
    _ensure_sqlite_column("campaigns", "data_json", "TEXT NOT NULL DEFAULT ''")


def _ensure_sqlite_column(table_name: str, column_name: str, column_def: str) -> None:
    """Add a SQLite column if it doesn't already exist."""
    with engine.begin() as conn:
        rows = conn.exec_driver_sql(f"PRAGMA table_info({table_name})").fetchall()
        existing = {row[1] for row in rows}  # row[1] is column name
        if column_name in existing:
            return
        conn.exec_driver_sql(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_def}")
