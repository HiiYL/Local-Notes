from __future__ import annotations

from typing import Optional
from dateutil import parser as dateparser


def parse_to_unix_ts(dt_str: str) -> Optional[int]:
    """Parse a date string (AppleScript modified string or ISO) to epoch seconds.

    Returns None if parsing fails.
    """
    if not dt_str:
        return None
    try:
        dt = dateparser.parse(dt_str)
        if not dt:
            return None
        # Assume local timezone if missing; dateutil may return naive
        if dt.tzinfo is None:
            from datetime import timezone
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except Exception:
        return None
