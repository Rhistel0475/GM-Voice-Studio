"""Add session memory table.

Revision ID: 004_session_memory
Revises: 003_campaign_documents
Create Date: 2026-03-16

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "004_session_memory"
down_revision: Union[str, Sequence[str], None] = "003_campaign_documents"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "session_memory",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("timestamp", sa.String(), nullable=False, server_default=sa.text("''")),
        sa.Column("event_type", sa.String(), nullable=False, server_default=sa.text("''")),
        sa.Column("npc_id", sa.String(), nullable=True),
        sa.Column("description", sa.Text(), nullable=False, server_default=sa.text("''")),
        if_not_exists=True,
    )
    op.create_index(
        "ix_session_memory_session_id",
        "session_memory",
        ["session_id"],
        unique=False,
        if_not_exists=True,
    )


def downgrade() -> None:
    op.drop_index("ix_session_memory_session_id", table_name="session_memory")
    op.drop_table("session_memory")
