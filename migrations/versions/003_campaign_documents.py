"""Add campaign document storage and chunk tables.

Revision ID: 003_campaign_documents
Revises: 002_npc_voice_session_events
Create Date: 2026-03-14

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "003_campaign_documents"
down_revision: Union[str, Sequence[str], None] = "002_npc_voice_session_events"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "campaign_documents",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("campaign_id", sa.Integer(), sa.ForeignKey("campaigns.id", ondelete="CASCADE"), nullable=False),
        sa.Column("filename", sa.String(), nullable=False, server_default=sa.text("''")),
        sa.Column("file_type", sa.String(), nullable=False, server_default=sa.text("''")),
        sa.Column("mime_type", sa.String(), nullable=False, server_default=sa.text("''")),
        sa.Column("storage_path", sa.String(), nullable=False, server_default=sa.text("''")),
        sa.Column("raw_text", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column("summary", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column("metadata_json", sa.Text(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("created_at", sa.String(), nullable=False, server_default=sa.text("''")),
        if_not_exists=True,
    )
    op.create_table(
        "campaign_document_chunks",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("document_id", sa.Integer(), sa.ForeignKey("campaign_documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("campaign_id", sa.Integer(), sa.ForeignKey("campaigns.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("token_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("text", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column("metadata_json", sa.Text(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("embedding_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.String(), nullable=False, server_default=sa.text("''")),
        if_not_exists=True,
    )
    op.create_index(
        "ix_campaign_document_chunks_campaign_id",
        "campaign_document_chunks",
        ["campaign_id"],
        unique=False,
        if_not_exists=True,
    )
    op.create_index(
        "ix_campaign_document_chunks_document_id",
        "campaign_document_chunks",
        ["document_id"],
        unique=False,
        if_not_exists=True,
    )


def downgrade() -> None:
    op.drop_index("ix_campaign_document_chunks_document_id", table_name="campaign_document_chunks")
    op.drop_index("ix_campaign_document_chunks_campaign_id", table_name="campaign_document_chunks")
    op.drop_table("campaign_document_chunks")
    op.drop_table("campaign_documents")
