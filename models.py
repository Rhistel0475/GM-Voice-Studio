"""
SQLAlchemy ORM models for GM Voice Studio campaign data.
Schema mirrors the JSON output of ai_full_parse() in ai_service.py.
"""
from sqlalchemy import Column, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from database import Base


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
