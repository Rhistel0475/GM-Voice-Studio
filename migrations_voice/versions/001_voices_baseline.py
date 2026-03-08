"""Baseline voices table (voice metadata).

Revision ID: 001_voices
Revises:
Create Date: Voice DB baseline

"""
from typing import Sequence, Union

from alembic import op

revision: str = "001_voices"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    is_sqlite = conn.dialect.name == "sqlite"
    if is_sqlite:
        op.execute("""
            CREATE TABLE IF NOT EXISTS voices (
                voice_id TEXT PRIMARY KEY,
                name TEXT NOT NULL DEFAULT '',
                consent_scope TEXT NOT NULL DEFAULT 'tts',
                created_at REAL NOT NULL,
                owner_id TEXT,
                faction TEXT
            )
        """)
    else:
        op.execute("""
            CREATE TABLE IF NOT EXISTS voices (
                voice_id TEXT PRIMARY KEY,
                name TEXT NOT NULL DEFAULT '',
                consent_scope TEXT NOT NULL DEFAULT 'tts',
                created_at DOUBLE PRECISION NOT NULL,
                owner_id TEXT,
                faction TEXT
            )
        """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS voices")
