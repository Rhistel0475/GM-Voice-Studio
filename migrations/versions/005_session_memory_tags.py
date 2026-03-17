"""Add tags to session_memory.

Revision ID: 005_session_memory_tags
Revises: 004_session_memory
Create Date: 2026-03-16

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "005_session_memory_tags"
down_revision: Union[str, Sequence[str], None] = "004_session_memory"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "session_memory",
        sa.Column("tags", sa.Text(), nullable=False, server_default=sa.text("'[]'")),
    )


def downgrade() -> None:
    op.drop_column("session_memory", "tags")
