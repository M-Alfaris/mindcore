"""
Timezone utilities for consistent datetime handling across Mindcore.

This module provides a single source of truth for all timezone operations,
ensuring consistent handling of:
- UTC timestamps (universal, stored in database)
- Local timestamps (user-facing, configurable)
- ISO 8601 parsing (various formats including 'Z' suffix)

Configuration:
    Set default timezone via environment variable or programmatically:
    - MINDCORE_TIMEZONE=America/New_York
    - set_default_timezone("Europe/London")

Database Storage Strategy:
    All timestamps are stored with TWO columns:
    - `created_at_utc` - Always UTC (for queries, sorting, consistency)
    - `created_at_local` - User's timezone (for display, debugging)
    - `timezone` - IANA timezone name (e.g., "America/New_York")
"""
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, Union
import os

# Try to import zoneinfo (Python 3.9+) or pytz as fallback
try:
    from zoneinfo import ZoneInfo
    _HAS_ZONEINFO = True
except ImportError:
    _HAS_ZONEINFO = False
    try:
        import pytz
        _HAS_PYTZ = True
    except ImportError:
        _HAS_PYTZ = False


# Default timezone (configurable)
_default_timezone: str = os.environ.get("MINDCORE_TIMEZONE", "UTC")


def set_default_timezone(tz_name: str) -> None:
    """
    Set the default timezone for the application.

    Args:
        tz_name: IANA timezone name (e.g., "America/New_York", "Europe/London", "UTC")
    """
    global _default_timezone
    # Validate timezone name
    get_timezone(tz_name)  # Raises if invalid
    _default_timezone = tz_name


def get_default_timezone() -> str:
    """Get the current default timezone name."""
    return _default_timezone


def get_timezone(tz_name: str) -> timezone:
    """
    Get a timezone object by IANA name.

    Args:
        tz_name: IANA timezone name (e.g., "America/New_York", "UTC")

    Returns:
        Timezone object

    Raises:
        ValueError: If timezone name is invalid
    """
    if tz_name == "UTC":
        return timezone.utc

    if _HAS_ZONEINFO:
        try:
            return ZoneInfo(tz_name)
        except Exception as e:
            raise ValueError(f"Invalid timezone: {tz_name}") from e
    elif _HAS_PYTZ:
        try:
            return pytz.timezone(tz_name)
        except Exception as e:
            raise ValueError(f"Invalid timezone: {tz_name}") from e
    else:
        # Fallback: only UTC supported without zoneinfo/pytz
        if tz_name != "UTC":
            raise ValueError(
                f"Timezone {tz_name} requires 'zoneinfo' (Python 3.9+) or 'pytz'. "
                "Install pytz: pip install pytz"
            )
        return timezone.utc


def utc_now() -> datetime:
    """
    Get current UTC timestamp (timezone-aware).

    Returns:
        Current datetime in UTC with tzinfo set
    """
    return datetime.now(timezone.utc)


def local_now(tz_name: Optional[str] = None) -> datetime:
    """
    Get current timestamp in specified timezone.

    Args:
        tz_name: Timezone name (defaults to MINDCORE_TIMEZONE)

    Returns:
        Current datetime in specified timezone
    """
    tz = get_timezone(tz_name or _default_timezone)
    return datetime.now(tz)


def to_utc(dt: datetime) -> datetime:
    """
    Convert datetime to UTC.

    Handles:
    - Naive datetimes (assumes local timezone)
    - Aware datetimes (converts to UTC)

    Args:
        dt: Datetime to convert

    Returns:
        UTC datetime (timezone-aware)
    """
    if dt.tzinfo is None:
        # Naive datetime - assume it's in default timezone
        local_tz = get_timezone(_default_timezone)
        if _HAS_PYTZ and hasattr(local_tz, 'localize'):
            dt = local_tz.localize(dt)
        else:
            dt = dt.replace(tzinfo=local_tz)

    return dt.astimezone(timezone.utc)


def to_local(dt: datetime, tz_name: Optional[str] = None) -> datetime:
    """
    Convert datetime to local timezone.

    Args:
        dt: Datetime to convert (can be naive or aware)
        tz_name: Target timezone (defaults to MINDCORE_TIMEZONE)

    Returns:
        Datetime in local timezone
    """
    # First ensure it's UTC
    utc_dt = to_utc(dt)

    # Convert to target timezone
    target_tz = get_timezone(tz_name or _default_timezone)
    return utc_dt.astimezone(target_tz)


def parse_iso(iso_string: str) -> datetime:
    """
    Parse ISO 8601 datetime string to UTC datetime.

    Handles various formats:
    - "2024-01-15T10:30:00Z" (Z suffix)
    - "2024-01-15T10:30:00+00:00" (offset)
    - "2024-01-15T10:30:00" (naive, assumed UTC)
    - "2024-01-15 10:30:00" (space separator)

    Args:
        iso_string: ISO 8601 formatted string

    Returns:
        UTC datetime (timezone-aware)
    """
    if not iso_string:
        return utc_now()

    # Normalize string
    normalized = iso_string.strip()

    # Handle 'Z' suffix (Zulu time = UTC)
    if normalized.endswith('Z'):
        normalized = normalized[:-1] + '+00:00'

    # Handle space separator (SQLite format)
    if ' ' in normalized and 'T' not in normalized:
        normalized = normalized.replace(' ', 'T')

    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        # Try parsing without timezone
        try:
            dt = datetime.fromisoformat(normalized.split('+')[0].split('-')[0])
        except ValueError:
            # Last resort: return current time
            return utc_now()

    return to_utc(dt)


def format_iso(dt: datetime, include_tz: bool = True) -> str:
    """
    Format datetime as ISO 8601 string.

    Args:
        dt: Datetime to format
        include_tz: Whether to include timezone offset

    Returns:
        ISO 8601 formatted string
    """
    utc_dt = to_utc(dt)
    if include_tz:
        return utc_dt.isoformat()
    else:
        return utc_dt.replace(tzinfo=None).isoformat()


def get_dual_timestamps(
    dt: Optional[datetime] = None,
    tz_name: Optional[str] = None
) -> Tuple[datetime, datetime, str]:
    """
    Get both UTC and local timestamps for database storage.

    This is the recommended way to store timestamps - always store both
    UTC (for consistency/queries) and local (for display/debugging).

    Args:
        dt: Source datetime (defaults to now)
        tz_name: Local timezone name (defaults to MINDCORE_TIMEZONE)

    Returns:
        Tuple of (utc_datetime, local_datetime, timezone_name)

    Example:
        >>> utc_dt, local_dt, tz_name = get_dual_timestamps()
        >>> # Store in database:
        >>> # created_at_utc = utc_dt
        >>> # created_at_local = local_dt
        >>> # timezone = tz_name
    """
    tz_name = tz_name or _default_timezone

    if dt is None:
        utc_dt = utc_now()
    else:
        utc_dt = to_utc(dt)

    local_dt = to_local(utc_dt, tz_name)

    return utc_dt, local_dt, tz_name


def normalize_for_comparison(dt: Union[datetime, str]) -> datetime:
    """
    Normalize a datetime for safe comparison.

    All comparisons should be done in UTC to avoid timezone issues.

    Args:
        dt: Datetime or ISO string to normalize

    Returns:
        UTC datetime (timezone-aware)
    """
    if isinstance(dt, str):
        return parse_iso(dt)
    return to_utc(dt)


def is_aware(dt: datetime) -> bool:
    """Check if datetime is timezone-aware."""
    return dt.tzinfo is not None and dt.tzinfo.utcoffset(dt) is not None


def ensure_aware(dt: datetime, assume_utc: bool = True) -> datetime:
    """
    Ensure datetime is timezone-aware.

    Args:
        dt: Datetime to check
        assume_utc: If naive, assume UTC (True) or local timezone (False)

    Returns:
        Timezone-aware datetime
    """
    if is_aware(dt):
        return dt

    if assume_utc:
        return dt.replace(tzinfo=timezone.utc)
    else:
        return to_utc(dt)  # Assumes local, converts to UTC


# Sorting key function for mixed datetime types
def sort_key(dt: Union[datetime, str, None]) -> datetime:
    """
    Get a sortable key from various datetime representations.

    Handles:
    - datetime objects (aware or naive)
    - ISO strings
    - None (returns epoch)

    Args:
        dt: Datetime, ISO string, or None

    Returns:
        UTC datetime for sorting
    """
    if dt is None:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)

    if isinstance(dt, str):
        return parse_iso(dt)

    return to_utc(dt)
