from __future__ import annotations

import time
from datetime import datetime, timedelta


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds_i = max(0, int(round(seconds)))
    hours, rem = divmod(seconds_i, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


def estimate_remaining(start_time: float, completed: int, total: int) -> tuple[float, float | None, datetime | None]:
    elapsed = time.perf_counter() - start_time
    if completed <= 0 or total <= 0:
        return elapsed, None, None
    remaining = (elapsed / completed) * max(total - completed, 0)
    finish = datetime.now() + timedelta(seconds=remaining)
    return elapsed, remaining, finish


def eta_line(label: str, start_time: float, completed: int, total: int) -> str:
    elapsed, remaining, finish = estimate_remaining(start_time, completed, total)
    pct = (completed / total * 100.0) if total else 0.0
    finish_text = finish.strftime("%Y-%m-%d %H:%M:%S") if finish is not None else "unknown"
    return (
        f"{label}: {completed}/{total} ({pct:5.1f}%) "
        f"| elapsed={format_duration(elapsed)} "
        f"| remaining={format_duration(remaining)} "
        f"| finish~{finish_text}"
    )
