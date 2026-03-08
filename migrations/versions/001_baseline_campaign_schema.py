"""Baseline campaign schema (campaigns, npcs, scenes, locations).

Revision ID: 001_baseline
Revises:
Create Date: Campaign DB (codm.db) baseline

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "001_baseline"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    is_sqlite = conn.dialect.name == "sqlite"

    op.create_table(
        "campaigns",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("title", sa.String(), nullable=False, server_default=sa.text("''")),
        sa.Column("summary", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column("data_json", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.PrimaryKeyConstraint("id"),
        if_not_exists=True,
    )
    op.create_table(
        "npcs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("campaign_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(), nullable=False, server_default=sa.text("''")),
        sa.Column("role", sa.String(), nullable=False, server_default=sa.text("''")),
        sa.Column("personality", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column("faction", sa.String(), nullable=False, server_default=sa.text("''")),
        sa.Column("description", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column("motivation", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column("secrets", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column("hp", sa.String(), nullable=False, server_default=sa.text("''")),
        sa.Column("ac", sa.Integer(), nullable=True),
        sa.Column("cr", sa.String(), nullable=False, server_default=sa.text("''")),
        sa.Column("image_url", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(["campaign_id"], ["campaigns.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        if_not_exists=True,
    )
    op.create_table(
        "scenes",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("campaign_id", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(), nullable=False, server_default=sa.text("''")),
        sa.Column("act", sa.String(), nullable=False, server_default=sa.text("''")),
        sa.Column("type", sa.String(), nullable=False, server_default=sa.text("''")),
        sa.Column("read_aloud", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column("difficulty", sa.String(), nullable=False, server_default=sa.text("''")),
        sa.Column("rewards", sa.String(), nullable=False, server_default=sa.text("''")),
        sa.Column("notes", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column("image_url", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(["campaign_id"], ["campaigns.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        if_not_exists=True,
    )
    op.create_table(
        "locations",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("campaign_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(), nullable=False, server_default=sa.text("''")),
        sa.Column("description", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column("image_url", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(["campaign_id"], ["campaigns.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        if_not_exists=True,
    )

    # Existing DBs created with create_all may lack data_json; add if missing (SQLite)
    if is_sqlite:
        result = conn.execute(sa.text("PRAGMA table_info(campaigns)"))
        columns = [row[1] for row in result]
        if "data_json" not in columns:
            op.add_column("campaigns", sa.Column("data_json", sa.Text(), nullable=False, server_default=sa.text("''")))


def downgrade() -> None:
    op.drop_table("locations")
    op.drop_table("scenes")
    op.drop_table("npcs")
    op.drop_table("campaigns")
