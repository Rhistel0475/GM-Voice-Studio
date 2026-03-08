"""
In-memory metrics for Prometheus-style /metrics endpoint.
Counters: tts_requests_total, clone_requests_total, errors_total.
Request duration: http_request_duration_seconds (summary: sum + count per path).
"""
import threading
from typing import List, Tuple

# Simple counters (name -> value)
_metrics: dict[str, int] = {
    "tts_requests_total": 0,
    "clone_requests_total": 0,
    "errors_total": 0,
}

# Request duration summary per path: path -> (sum_seconds, count)
_lock = threading.Lock()
_duration_sum: dict[str, float] = {}
_duration_count: dict[str, int] = {}


def increment(name: str, value: int = 1) -> None:
    if name in _metrics:
        _metrics[name] += value


def record_request_duration(path: str, duration_seconds: float) -> None:
    """Record request duration for a given path (for Prometheus summary)."""
    with _lock:
        _duration_sum[path] = _duration_sum.get(path, 0.0) + duration_seconds
        _duration_count[path] = _duration_count.get(path, 0) + 1


def get_all() -> List[Tuple[str, int]]:
    return list(_metrics.items())


def prometheus_text() -> str:
    """Return metrics in Prometheus exposition format (text)."""
    lines = []
    for name, value in _metrics.items():
        lines.append(f"# HELP {name} Counter")
        lines.append(f"# TYPE {name} counter")
        lines.append(f"{name} {value}")
    lines.append("# HELP http_request_duration_seconds Request duration by path (summary)")
    lines.append("# TYPE http_request_duration_seconds summary")
    with _lock:
        for path in sorted(_duration_sum.keys()):
            p = path or "/"
            sum_val = _duration_sum[path]
            count_val = _duration_count[path]
            lines.append(f'http_request_duration_seconds_sum{{path="{p}"}} {sum_val}')
            lines.append(f'http_request_duration_seconds_count{{path="{p}"}} {count_val}')
    return "\n".join(lines) + "\n"
