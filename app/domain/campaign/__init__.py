"""
Campaign domain: adventure parsing, campaign persistence, NPCs/scenes/locations.

This package intentionally avoids importing the campaign repository at module load
time so submodules like `app.domain.campaign.systems` can be imported without
creating circular imports.
"""
from __future__ import annotations

from typing import Any

from app.domain.campaign.systems import (
    DEFAULT_CAMPAIGN_SYSTEM_ID,
    get_campaign_system_preset,
    list_campaign_system_presets,
    normalize_campaign_system,
)


def _campaign_repository():
    from app.repositories import campaign_repository

    return campaign_repository


def list_all(*args: Any, **kwargs: Any):
    return _campaign_repository().list_all(*args, **kwargs)


def get_by_id(*args: Any, **kwargs: Any):
    return _campaign_repository().get_by_id(*args, **kwargs)


def delete(*args: Any, **kwargs: Any):
    return _campaign_repository().delete(*args, **kwargs)


def create_from_parse_result(*args: Any, **kwargs: Any):
    return _campaign_repository().create_from_parse_result(*args, **kwargs)


def __getattr__(name: str):
    if name == "campaign_repository":
        return _campaign_repository()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "campaign_repository",
    "list_all",
    "get_by_id",
    "delete",
    "create_from_parse_result",
    "DEFAULT_CAMPAIGN_SYSTEM_ID",
    "normalize_campaign_system",
    "get_campaign_system_preset",
    "list_campaign_system_presets",
]
