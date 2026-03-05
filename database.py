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
