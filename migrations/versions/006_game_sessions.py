"""Add game_sessions for active scene index per client session.

Revision ID: 006_game_sessions
Revises: 005_session_memory_tags
Create Date: 2026-03-16

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "006_game_sessions"
down_revision: Union[str, Sequence[str], None] = "005_session_memory_tags"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "game_sessions",
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("campaign_id", sa.Integer(), nullable=True),
        sa.Column("active_scene_index", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("updated_at", sa.String(), nullable=False, server_default=sa.text("''")),
        sa.ForeignKeyConstraint(["campaign_id"], ["campaigns.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("session_id"),
        if_not_exists=True,
    )


def downgrade() -> None:
    op.drop_table("game_sessions")
