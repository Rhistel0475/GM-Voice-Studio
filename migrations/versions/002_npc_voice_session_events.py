"""Add voice_id to npcs and session_events table.

Revision ID: 002_npc_voice_session_events
Revises: 001_baseline
Create Date: 2026-03-11

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "002_npc_voice_session_events"
down_revision: Union[str, Sequence[str], None] = "001_baseline"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    is_sqlite = conn.dialect.name == "sqlite"

    # Add voice_id to npcs (idempotent for SQLite)
    if is_sqlite:
        result = conn.execute(sa.text("PRAGMA table_info(npcs)"))
        columns = [row[1] for row in result]
        if "voice_id" not in columns:
            op.add_column("npcs", sa.Column("voice_id", sa.String(), nullable=True))
    else:
        op.add_column("npcs", sa.Column("voice_id", sa.String(), nullable=True))

    # Create session_events table
    op.create_table(
        "session_events",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("campaign_id", sa.Integer(), sa.ForeignKey("campaigns.id", ondelete="CASCADE"), nullable=False),
        sa.Column("scene_id", sa.String(), nullable=True),
        sa.Column("session_id", sa.String(), nullable=True),
        sa.Column("type", sa.String(), nullable=False, server_default=sa.text("'assistant'")),
        sa.Column("text", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column("created_at", sa.String(), nullable=False, server_default=sa.text("''")),
        if_not_exists=True,
    )


def downgrade() -> None:
    op.drop_table("session_events")
    # SQLite does not support DROP COLUMN natively; skip for downgrade
