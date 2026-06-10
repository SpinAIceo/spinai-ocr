"""Simple in-memory monthly-quota rate limiter for freemium tier.

Per-identity (API key or IP): 50 successful /ocr calls per calendar month.
Resets at UTC month boundary.

For production B2B, replace this with Redis-backed or database-backed quota
tracking (this one loses state on server restart). Good enough for MVP +
proof-of-concept demos.

Usage:
    from spinaiocr.serve.ratelimit import check_monthly_quota, commit_monthly_call
    check_monthly_quota(identity="1.2.3.4", limit=50)  # raises 429 on overage
    ... do the work ...
    commit_monthly_call(identity="1.2.3.4")            # only count successes
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone

from fastapi import HTTPException

# (identity, year-month) -> count used
_COUNTS: dict[tuple[str, str], int] = {}
_LOCK = threading.Lock()


def _month_key(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return now.strftime("%Y-%m")


def check_monthly_quota(identity: str, limit: int = 50) -> int:
    """Raise 429 if (identity, this month) is at or over limit. No-op otherwise.

    Does NOT increment — call commit_monthly_call() only on success. This
    ensures malformed requests (413/400) don't burn quota.

    Limit choice (50 free/month): aggressive freemium gating for B2B pivot.
    Enough to try-before-buy, not enough to build a free product on top of.
    Paid tier lifts this via `_QUOTA_OVERRIDE[identity]`.

    Returns the current used count (pre-commit).
    """
    key = (identity or "anon", _month_key())
    override = _QUOTA_OVERRIDE.get(identity)
    eff = override if override is not None else limit
    with _LOCK:
        used = _COUNTS.get(key, 0)
        if used >= eff:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "monthly_quota_exceeded",
                    "used": used,
                    "limit": eff,
                    "identity": identity,
                    "reset_utc": f"{key[1]}-01 next month 00:00 UTC",
                    "upgrade": "contact hello@spinai for paid tier",
                },
            )
        return used


def commit_monthly_call(identity: str) -> int:
    """Record one successful /ocr call for (identity, this month)."""
    key = (identity or "anon", _month_key())
    with _LOCK:
        _COUNTS[key] = _COUNTS.get(key, 0) + 1
        return _COUNTS[key]


def get_usage(identity: str) -> dict:
    """Inspect current usage (for /quota endpoint, client transparency)."""
    key = (identity or "anon", _month_key())
    override = _QUOTA_OVERRIDE.get(identity)
    limit = override if override is not None else _DEFAULT_LIMIT
    with _LOCK:
        used = _COUNTS.get(key, 0)
    return {
        "identity": identity,
        "period": key[1],
        "used": used,
        "limit": limit,
        "remaining": max(0, limit - used),
    }


# Identities with paid-tier overrides (key=identity, value=new monthly limit)
# Populated at runtime by admin endpoint / config. In-memory for MVP.
_QUOTA_OVERRIDE: dict[str, int] = {}
_DEFAULT_LIMIT = 50


def set_quota_override(identity: str, limit: int | None) -> None:
    """Set a per-identity monthly limit. limit=None removes override."""
    with _LOCK:
        if limit is None:
            _QUOTA_OVERRIDE.pop(identity, None)
        else:
            _QUOTA_OVERRIDE[identity] = limit


def reset_all() -> None:
    """Wipe counters (used only by tests)."""
    with _LOCK:
        _COUNTS.clear()
        _QUOTA_OVERRIDE.clear()
