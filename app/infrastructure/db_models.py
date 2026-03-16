"""
SQLAlchemy ORM models for GM Voice Studio campaign data.
Schema mirrors the JSON output of ai_full_parse() in ai_service.py.
"""
from sqlalchemy import Column, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from app.infrastructure.database import Base


class Campaign(Base):
    __tablename__ = "campaigns"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String, nullable=False, default="")
    summary = Column(Text, nullable=False, default="")
    # Canonical serialized campaign payload (party, reveals, items, scene links, images, etc.).
    data_json = Column(Text, nullable=False, default="")

    # One campaign → many NPCs, Scenes, Locations (cascade delete-orphan)
    npcs = relationship("NPC", back_populates="campaign", cascade="all, delete-orphan")
    scenes = relationship("Scene", back_populates="campaign", cascade="all, delete-orphan")
    locations = relationship("Location", back_populates="campaign", cascade="all, delete-orphan")
    session_events = relationship("SessionEvent", back_populates="campaign", cascade="all, delete-orphan")
    documents = relationship("CampaignDocument", back_populates="campaign", cascade="all, delete-orphan")


class NPC(Base):
    __tablename__ = "npcs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    campaign_id = Column(Integer, ForeignKey("campaigns.id", ondelete="CASCADE"), nullable=False)

    name = Column(String, nullable=False, default="")
    role = Column(String, nullable=False, default="")         # villain|ally|quest-giver|neutral
    personality = Column(Text, nullable=False, default="")
    faction = Column(String, nullable=False, default="")
    description = Column(Text, nullable=False, default="")
    motivation = Column(Text, nullable=False, default="")
    secrets = Column(Text, nullable=False, default="")
    hp = Column(String, nullable=False, default="")           # e.g. "45" or "3d8"
    ac = Column(Integer, nullable=True)                       # armor class
    cr = Column(String, nullable=False, default="")           # e.g. "CR 3"
    image_url = Column(String, nullable=True)
    voice_id = Column(String, nullable=True)                  # assigned TTS voice

    campaign = relationship("Campaign", back_populates="npcs")


class Scene(Base):
    __tablename__ = "scenes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    campaign_id = Column(Integer, ForeignKey("campaigns.id", ondelete="CASCADE"), nullable=False)

    title = Column(String, nullable=False, default="")
    act = Column(String, nullable=False, default="")
    type = Column(String, nullable=False, default="")         # combat|social|exploration|mystery
    read_aloud = Column(Text, nullable=False, default="")
    difficulty = Column(String, nullable=False, default="")   # easy|medium|hard|deadly|none
    rewards = Column(String, nullable=False, default="")
    notes = Column(Text, nullable=False, default="")
    image_url = Column(String, nullable=True)

    campaign = relationship("Campaign", back_populates="scenes")


class Location(Base):
    __tablename__ = "locations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    campaign_id = Column(Integer, ForeignKey("campaigns.id", ondelete="CASCADE"), nullable=False)

    name = Column(String, nullable=False, default="")
    description = Column(Text, nullable=False, default="")
    image_url = Column(String, nullable=True)

    campaign = relationship("Campaign", back_populates="locations")


class SessionEvent(Base):
    __tablename__ = "session_events"

    id = Column(String, primary_key=True)
    campaign_id = Column(Integer, ForeignKey("campaigns.id", ondelete="CASCADE"), nullable=False)
    scene_id = Column(String, nullable=True)
    session_id = Column(String, nullable=True)
    type = Column(String, nullable=False, default="assistant")
    text = Column(Text, nullable=False, default="")
    created_at = Column(String, nullable=False, default="")

    campaign = relationship("Campaign", back_populates="session_events")


class SessionMemory(Base):
    __tablename__ = "session_memory"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False, index=True)
    timestamp = Column(String, nullable=False, default="")
    event_type = Column(String, nullable=False, default="")
    npc_id = Column(String, nullable=True)
    description = Column(Text, nullable=False, default="")


class CampaignDocument(Base):
    __tablename__ = "campaign_documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    campaign_id = Column(Integer, ForeignKey("campaigns.id", ondelete="CASCADE"), nullable=False)
    filename = Column(String, nullable=False, default="")
    file_type = Column(String, nullable=False, default="")
    mime_type = Column(String, nullable=False, default="")
    storage_path = Column(String, nullable=False, default="")
    raw_text = Column(Text, nullable=False, default="")
    summary = Column(Text, nullable=False, default="")
    metadata_json = Column(Text, nullable=False, default="")
    chunk_count = Column(Integer, nullable=False, default=0)
    created_at = Column(String, nullable=False, default="")

    campaign = relationship("Campaign", back_populates="documents")
    chunks = relationship("CampaignDocumentChunk", back_populates="document", cascade="all, delete-orphan")


class CampaignDocumentChunk(Base):
    __tablename__ = "campaign_document_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("campaign_documents.id", ondelete="CASCADE"), nullable=False)
    campaign_id = Column(Integer, ForeignKey("campaigns.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False, default=0)
    token_count = Column(Integer, nullable=False, default=0)
    text = Column(Text, nullable=False, default="")
    metadata_json = Column(Text, nullable=False, default="")
    embedding_json = Column(Text, nullable=True)
    created_at = Column(String, nullable=False, default="")

    document = relationship("CampaignDocument", back_populates="chunks")
