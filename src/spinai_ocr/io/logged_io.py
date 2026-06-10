"""Core logged I/O helpers.

Every write path goes through `_commit(...)` which:
    1. logs `io.begin` with target path
    2. writes to `<path>.tmp`
    3. atomically renames into place
    4. computes sha256 on request
    5. logs `io.commit` with size/sha/elapsed_ms

On any exception the `.tmp` file is removed and `io.failed` is logged with
the full traceback plus a crash dump.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

from spinai_ocr.log import capture_crashes, get_logger, log_span

log = get_logger("spinai_ocr.io")

# How often to emit progress for long writes (bytes).
PROGRESS_CHUNK = 4 * 1024 * 1024


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    existed = p.exists()
    p.mkdir(parents=True, exist_ok=True)
    log.debug("io.ensure_dir path=%s existed=%s", p, existed,
              extra={"op": "ensure_dir", "path": str(p), "existed": existed})
    return p


def sha256_of_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


@contextmanager
def _commit(path: Path, op: str, sha256: bool, extra: dict | None = None):
    """Atomic write context. yields the tmp-path for the caller to populate."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp") if path.suffix else path.with_name(path.name + ".tmp")
    started = time.perf_counter()
    payload = {"op": op, "path": str(path), **(extra or {})}
    log.info("io.begin op=%s path=%s", op, path, extra=payload)

    try:
        with capture_crashes(f"io.{op}", extra=payload, reraise=True):
            yield tmp
            # atomic rename
            os.replace(tmp, path)
    except BaseException:
        # remove tmp if it still exists
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:  # noqa: BLE001
            pass
        log.error(
            "io.failed op=%s path=%s",
            op, path,
            extra={**payload, "event": "failed"},
        )
        raise

    size = path.stat().st_size
    elapsed_ms = (time.perf_counter() - started) * 1000
    done_payload = {**payload, "size_bytes": size, "elapsed_ms": elapsed_ms, "event": "commit"}
    if sha256:
        done_payload["sha256"] = sha256_of_file(path)
    log.info(
        "io.commit op=%s path=%s size=%s elapsed_ms=%.1f%s",
        op,
        path,
        _fmt_bytes(size),
        elapsed_ms,
        f" sha256={done_payload['sha256'][:12]}..." if sha256 else "",
        extra=done_payload,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def atomic_write_bytes(path: str | Path, data: bytes, sha256: bool = False, op: str = "write_bytes") -> Path:
    p = Path(path)
    with _commit(p, op, sha256=sha256, extra={"n_bytes": len(data)}) as tmp:
        tmp.write_bytes(data)
    return p


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8", sha256: bool = False, op: str = "write_text") -> Path:
    p = Path(path)
    with _commit(p, op, sha256=sha256, extra={"n_chars": len(text)}) as tmp:
        tmp.write_text(text, encoding=encoding)
    return p


def save_json(path: str | Path, obj: Any, *, sha256: bool = False, indent: int = 2) -> Path:
    p = Path(path)
    text = json.dumps(obj, ensure_ascii=False, indent=indent, default=str)
    return atomic_write_text(p, text, sha256=sha256, op="save_json")


def save_jsonl_append(path: str | Path, records: Iterable[dict]) -> int:
    """JSONL append is intrinsically non-atomic; log per-batch stats."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    n = 0
    before = p.stat().st_size if p.exists() else 0
    log.info("io.begin op=jsonl_append path=%s", p,
             extra={"op": "jsonl_append", "path": str(p), "size_before": before})
    try:
        with p.open("a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                n += 1
    except BaseException:
        log.error("io.failed op=jsonl_append path=%s n_written=%d", p, n,
                  extra={"op": "jsonl_append", "path": str(p), "n_written": n},
                  exc_info=True)
        raise
    after = p.stat().st_size
    elapsed_ms = (time.perf_counter() - started) * 1000
    log.info(
        "io.commit op=jsonl_append path=%s records=%d delta=%s elapsed_ms=%.1f",
        p, n, _fmt_bytes(after - before), elapsed_ms,
        extra={
            "op": "jsonl_append", "path": str(p), "n_records": n,
            "size_before": before, "size_after": after,
            "delta_bytes": after - before, "elapsed_ms": elapsed_ms,
        },
    )
    return n


def save_image(path: str | Path, pil_or_array, *, format: str | None = None, sha256: bool = False) -> Path:
    """PIL.Image or HxWxC numpy array — save to disk with logging."""
    import numpy as np
    from PIL import Image

    p = Path(path)
    if isinstance(pil_or_array, np.ndarray):
        img = Image.fromarray(pil_or_array)
    else:
        img = pil_or_array

    with _commit(p, "save_image", sha256=sha256,
                 extra={"width": img.width, "height": img.height, "mode": img.mode}) as tmp:
        save_kwargs = {}
        if format:
            save_kwargs["format"] = format
        img.save(tmp, **save_kwargs)
    return p


def save_torch_ckpt(path: str | Path, state: dict, *, sha256: bool = True, op: str = "save_ckpt") -> Path:
    """torch.save wrapper with atomic rename + sha256.

    Prefer this over `torch.save(model.state_dict(), path)` so every
    checkpoint ends up with a log entry (path / size / sha256 / elapsed).
    """
    import torch  # type: ignore

    p = Path(path)
    n_keys = len(state.get("state_dict", state)) if isinstance(state, dict) else None
    with _commit(p, op, sha256=sha256, extra={"n_state_keys": n_keys}) as tmp:
        torch.save(state, tmp)
    return p


def download_to_file(url: str, dest: str | Path, *, chunk: int = PROGRESS_CHUNK, sha256: bool = False, timeout: float = 60.0) -> Path:
    """HTTP GET → file with periodic progress logs."""
    p = Path(dest)
    p.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    total_read = 0
    log.info("io.begin op=download url=%s dest=%s", url, p,
             extra={"op": "download", "url": url, "path": str(p)})
    tmp = p.with_suffix(p.suffix + ".tmp") if p.suffix else p.with_name(p.name + ".tmp")

    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp, tmp.open("wb") as f:
            total = resp.headers.get("Content-Length")
            total_int = int(total) if total and total.isdigit() else None
            next_progress = chunk
            while True:
                block = resp.read(chunk)
                if not block:
                    break
                f.write(block)
                total_read += len(block)
                if total_read >= next_progress:
                    pct = f"{(total_read / total_int * 100):.1f}%" if total_int else "?"
                    log.debug(
                        "io.progress op=download url=%s read=%s/%s (%s)",
                        url, _fmt_bytes(total_read), _fmt_bytes(total_int) if total_int else "?", pct,
                        extra={"op": "download", "bytes_read": total_read, "total": total_int},
                    )
                    next_progress += chunk
        os.replace(tmp, p)
    except BaseException:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:  # noqa: BLE001
            pass
        log.error("io.failed op=download url=%s dest=%s bytes=%d", url, p, total_read,
                  exc_info=True, extra={"op": "download", "url": url, "path": str(p)})
        raise

    size = p.stat().st_size
    elapsed_ms = (time.perf_counter() - started) * 1000
    speed = (total_read / 1e6) / (elapsed_ms / 1000) if elapsed_ms else 0.0
    done = {
        "op": "download", "url": url, "path": str(p), "size_bytes": size,
        "elapsed_ms": elapsed_ms, "speed_MBps": speed, "event": "commit",
    }
    if sha256:
        done["sha256"] = sha256_of_file(p)
    log.info(
        "io.commit op=download url=%s dest=%s size=%s elapsed_ms=%.1f speed=%.2f MB/s",
        url, p, _fmt_bytes(size), elapsed_ms, speed, extra=done,
    )
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_bytes(n) -> str:
    if n is None:
        return "?"
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.2f}{unit}" if unit != "B" else f"{int(n)}B"
        n /= 1024
    return f"{n:.2f}PB"
