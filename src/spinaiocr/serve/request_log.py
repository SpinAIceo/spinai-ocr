"""Append-only per-request log for /ocr calls — enables retrospective
model-performance analysis (CER, latency, tier comparison, confidence
distribution) without re-running live traffic.

One JSON object per line in ``$SPINAI_REQUEST_LOG_DIR/ocr_requests.jsonl``
(default: ``logs/ocr_requests.jsonl``). Logging never raises — /ocr must
never 500 because of telemetry.

Privacy: client identities are hashed (sha256 first 12 chars), image
bytes are NOT stored — only size, dimensions, and a 16-char image sha256
so the same image across requests can be correlated.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOG_DIR = Path(os.environ.get("SPINAI_REQUEST_LOG_DIR", "logs"))
_LOG_FILE = _LOG_DIR / "ocr_requests.jsonl"
_LOG_LOCK = threading.Lock()

# Cap the on-disk file so a long-lived container doesn't eat the root FS.
# When the file crosses this threshold we rotate to ocr_requests.jsonl.1
# (overwriting any previous rotation). Railway's default instance has a few
# GB of writable disk; 50 MB of JSONL is ~200k requests.
_MAX_BYTES = int(os.environ.get("SPINAI_REQUEST_LOG_MAX_BYTES", 50 * 1024 * 1024))


def anon_identity(identity: str) -> str:
    """Map ``ip:1.2.3.4`` or ``key:xxx`` → short stable hash so logs can
    distinguish callers without leaking raw IPs."""
    h = hashlib.sha256(identity.encode()).hexdigest()[:12]
    kind = identity.split(":", 1)[0] if ":" in identity else "unk"
    return f"{kind}:{h}"


def image_fingerprint(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _rotate_if_needed() -> None:
    try:
        if _LOG_FILE.exists() and _LOG_FILE.stat().st_size > _MAX_BYTES:
            rotated = _LOG_FILE.with_suffix(".jsonl.1")
            if rotated.exists():
                rotated.unlink()
            _LOG_FILE.rename(rotated)
    except Exception:
        pass


def log_ocr_request(
    *,
    req_id: str,
    identity: str,
    params: dict[str, Any],
    image: dict[str, Any],
    result: dict[str, Any],
    timing_ms: dict[str, Any],
    error: str | None = None,
) -> None:
    """Append a single OCR call as one JSON line. Never raises."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "req_id": req_id,
        "identity": anon_identity(identity),
        "params": params,
        "image": image,
        "result": result,
        "timing_ms": timing_ms,
    }
    if error is not None:
        entry["error"] = error
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        with _LOG_LOCK:
            _rotate_if_needed()
            with _LOG_FILE.open("a", encoding="utf-8") as f:
                f.write(line)
    except Exception:
        # Telemetry must never break the request path.
        pass


def read_recent(limit: int = 100) -> list[dict[str, Any]]:
    """Return last ``limit`` JSON entries (newest-last). Silently tolerates
    rotation and partial writes."""
    if not _LOG_FILE.exists():
        return []
    try:
        with _LOG_FILE.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return []
    # Cheap: read all, take tail. For 50MB cap this is ~200k lines max —
    # well under a GB of Python strings. If the file grows further we'd
    # switch to a streaming read backwards.
    out: list[dict[str, Any]] = []
    for raw in lines[-limit:]:
        try:
            out.append(json.loads(raw))
        except Exception:
            continue
    return out


def aggregate(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarise entries by (tier, decode). Quick-n-dirty percentiles."""
    if not entries:
        return {"n_total": 0, "by_tier_decode": {}}
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for e in entries:
        params = e.get("params", {})
        key = (params.get("tier", "?"), params.get("decode", "?"))
        buckets.setdefault(key, []).append(e)

    def _pct(xs: list[float], p: float) -> float:
        if not xs:
            return 0.0
        xs = sorted(xs)
        k = max(0, min(len(xs) - 1, int(round((len(xs) - 1) * p))))
        return xs[k]

    summary: dict[str, Any] = {}
    for (tier, decode), rows in buckets.items():
        confs = [r.get("result", {}).get("avg_conf", 0.0) for r in rows]
        times = [r.get("timing_ms", {}).get("total", 0.0) for r in rows]
        unks = [r.get("result", {}).get("total_unk", 0) for r in rows]
        chars = [r.get("result", {}).get("total_chars", 0) for r in rows]
        n = len(rows)
        summary[f"{tier}/{decode}"] = {
            "n": n,
            "avg_conf_mean": round(sum(confs) / n, 4) if n else 0.0,
            "avg_conf_p50": round(_pct(confs, 0.5), 4),
            "latency_ms_p50": round(_pct(times, 0.5), 1),
            "latency_ms_p95": round(_pct(times, 0.95), 1),
            "unk_rate": round(sum(1 for u in unks if u > 0) / n, 3) if n else 0.0,
            "total_chars_mean": round(sum(chars) / n, 1) if n else 0.0,
        }
    return {"n_total": len(entries), "by_tier_decode": summary}
