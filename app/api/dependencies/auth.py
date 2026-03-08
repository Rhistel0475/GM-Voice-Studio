"""Shared FastAPI dependencies: auth, rate limiting, abuse check."""
import time
from typing import Optional

from fastapi import HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import (
    ABUSE_CLONE_PER_IP_PER_HOUR,
    API_KEYS,
    RATE_LIMIT_GLOBAL,
    REQUIRE_API_KEY,
)

limiter = Limiter(
    key_func=get_remote_address,
    application_limits=[RATE_LIMIT_GLOBAL] if RATE_LIMIT_GLOBAL else [],
)

# Abuse: clone count per IP (in-memory, last hour)
_clone_times_by_ip: dict[str, list[float]] = {}


async def verify_api_key(request: Request) -> None:
    """Optional API key verification (when REQUIRE_API_KEY and API_KEYS are set)."""
    if not REQUIRE_API_KEY or not API_KEYS:
        return
    key = (
        request.headers.get("X-API-Key")
        or (request.headers.get("Authorization") or "").replace("Bearer ", "").strip()
    )
    if not key or key not in API_KEYS:
        raise HTTPException(401, "Invalid or missing API key")


def get_owner_id(request: Request) -> Optional[str]:
    """Resolve owner from request: valid API key or None. Used for per-user voice scoping when DB is set."""
    if not API_KEYS:
        return None
    key = (
        request.headers.get("X-API-Key")
        or (request.headers.get("Authorization") or "").replace("Bearer ", "").strip()
    )
    return key if key in API_KEYS else None


def check_abuse_clone(ip: str) -> None:
    """Raise 429 if too many clone requests from this IP in the last hour."""
    if ABUSE_CLONE_PER_IP_PER_HOUR <= 0:
        return
    now = time.time()
    cutoff = now - 3600
    if ip not in _clone_times_by_ip:
        _clone_times_by_ip[ip] = []
    times = _clone_times_by_ip[ip]
    times.append(now)
    times[:] = [t for t in times if t > cutoff]
    if len(times) > ABUSE_CLONE_PER_IP_PER_HOUR:
        raise HTTPException(
            429, "Too many voice clones from this IP; try again later"
        )
