"""
Stage 1: Normalize document text and preserve section boundaries.
Cleans whitespace and encoding without losing heading markers.
"""
import re
from typing import Optional


def normalize_text(text: str, max_chars: Optional[int] = None) -> str:
    """
    Clean adventure text for parsing: normalize whitespace, fix common PDF artifacts,
    preserve heading markers (#, **Bold**, Chapter N, etc.).

    Args:
        text: Raw document text.
        max_chars: Optional cap (e.g. MAX_ADVENTURE_CHARS). Truncate at word boundary.

    Returns:
        Normalized string.
    """
    if not text or not text.strip():
        return ""

    # Replace common problematic chars
    out = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse multiple blank lines to double newline (preserve paragraph breaks)
    out = re.sub(r"\n{3,}", "\n\n", out)
    # Normalize spaces: multiple spaces/tabs to single space, but keep single newlines
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r" *\n *", "\n", out)
    out = out.strip()

    if max_chars and len(out) > max_chars:
        # Truncate at word boundary
        out = out[:max_chars].rsplit(" ", 1)[0]
        if not out:
            out = text[:max_chars]

    return out
