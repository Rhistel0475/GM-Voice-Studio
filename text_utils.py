"""
Text splitting for long-form TTS (script to narration).
"""
import re
from typing import List

# Limits to avoid timeouts (sync request)
MAX_TOTAL_CHARS = 5000
MAX_CHUNKS = 15


def split_for_tts(
    text: str,
    chunk_by: str = "sentence",
    max_chars: int = 500,
) -> List[str]:
    """
    Split text into chunks for TTS. Returns list of non-empty strings.
    chunk_by: "sentence" | "paragraph" | "fixed"
    max_chars: used when chunk_by is "fixed" (break at word boundary).
    Enforces MAX_TOTAL_CHARS and MAX_CHUNKS.
    """
    text = (text or "").strip()
    if not text:
        return []

    if len(text) > MAX_TOTAL_CHARS:
        text = text[:MAX_TOTAL_CHARS]

    chunks: List[str] = []

    if chunk_by == "paragraph":
        raw = [p.strip() for p in text.split("\n\n") if p.strip()]
        for p in raw:
            if len(chunks) >= MAX_CHUNKS:
                break
            if len(p) > MAX_TOTAL_CHARS:
                p = p[:MAX_TOTAL_CHARS]
            chunks.append(p)
    elif chunk_by == "fixed":
        max_chars = max(50, min(max_chars, 1500))
        remaining = text
        while remaining and len(chunks) < MAX_CHUNKS:
            if len(remaining) <= max_chars:
                chunks.append(remaining.strip())
                break
            segment = remaining[:max_chars]
            last_space = segment.rfind(" ")
            if last_space > max_chars // 2:
                cut = last_space + 1
            else:
                cut = max_chars
            chunks.append(remaining[:cut].strip())
            remaining = remaining[cut:].strip()
    else:
        # sentence: split on . ! ? followed by space or end
        raw = re.split(r'(?<=[.!?])\s+', text)
        current = ""
        for s in raw:
            s = s.strip()
            if not s:
                continue
            if len(current) + len(s) + 1 <= max_chars and len(chunks) < MAX_CHUNKS:
                current = (current + " " + s).strip() if current else s
            else:
                if current:
                    chunks.append(current)
                    if len(chunks) >= MAX_CHUNKS:
                        break
                current = s[:max_chars] if len(s) > max_chars else s
        if current and len(chunks) < MAX_CHUNKS:
            chunks.append(current)

    return chunks[:MAX_CHUNKS]
