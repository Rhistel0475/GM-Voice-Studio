"""
Structured logging for JSON output (e.g. for production log aggregation).
Set LOG_JSON=1 to enable JSON format; otherwise use default format.
"""
import json
import logging
import os
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    """Format log records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        log = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "request_path"):
            log["path"] = getattr(record, "request_path", None)
        if hasattr(record, "status_code"):
            log["status_code"] = getattr(record, "status_code", None)
        if hasattr(record, "duration_seconds"):
            log["duration_seconds"] = getattr(record, "duration_seconds", None)
        if hasattr(record, "voice_id") and getattr(record, "voice_id", None) is not None:
            log["voice_id"] = getattr(record, "voice_id", None)
        if hasattr(record, "job_id") and getattr(record, "job_id", None) is not None:
            log["job_id"] = getattr(record, "job_id", None)
        return json.dumps(log)


def configure_logging() -> None:
    """Configure root logger: JSON if LOG_JSON=1, else default."""
    use_json = os.environ.get("LOG_JSON", "").strip() in ("1", "true", "yes")
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JsonFormatter() if use_json else logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s"
        ))
        root.addHandler(handler)
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    root.setLevel(getattr(logging, level, logging.INFO))
