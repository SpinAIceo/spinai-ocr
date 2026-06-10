"""Hard-negative queue for the production data flywheel.

When production /ocr returns a result with LOW mean confidence (below a
configurable threshold), we persist the request for later re-labelling by
the teacher ensemble.

Pipeline (user-proposed _52):
    1. Serve: low-conf /ocr → append JSONL entry + save image.
    2. Periodic job: send queued images to consensus teachers
       (Qwen-VL + Paddle + EasyOCR) — same path as distill.
    3. Gate the teacher outputs (agreement, min_top1, etc.).
    4. Retrain student on new (image, consensus_text) pairs.
    5. Result: the longer the system runs in production, the more robust
       it gets — especially on the idiosyncratic shapes that matter to
       the paying customers.

Storage:
    data/hard_negatives/queue.jsonl          one line per queued request
    data/hard_negatives/images/<uuid>.jpg    the image as received

Disabled by default via env var so CI / dev runs don't disk-fill.
Set SPINAI_HN_QUEUE=1 (or any truthy) to enable in prod.
"""
from __future__ import annotations

import io
import json
import os
import random
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

_QUEUE_ROOT = Path(os.environ.get("SPINAI_HN_DIR", "data/hard_negatives"))
_ENABLED = os.environ.get("SPINAI_HN_QUEUE", "").lower() in {"1", "true", "yes"}
_THRESHOLD = float(os.environ.get("SPINAI_HN_THRESHOLD", "0.60"))
# 2026-06-08: conf is miscalibrated — the model is confidently wrong (conf
# 0.92 on garbage), so the conf<_THRESHOLD gate misses ~80% of real failures.
# Add a conf-blind random sample of ALL requests so high-conf-wrong failures
# also enter the queue for offline (human/disagreement) labelling. 0 = off.
_SAMPLE_RATE = float(os.environ.get("SPINAI_HN_SAMPLE_RATE", "0"))
_MAX_QUEUE_MB = int(os.environ.get("SPINAI_HN_MAX_MB", "500"))

_LOCK = threading.Lock()


def _queue_size_mb() -> float:
    img_dir = _QUEUE_ROOT / "images"
    if not img_dir.exists():
        return 0.0
    total = sum(p.stat().st_size for p in img_dir.iterdir() if p.is_file())
    return total / (1024 * 1024)


def maybe_queue_hard_negative(
    img: Image.Image,
    lines: list,  # list of OCRLine-like objects with .confidence, .text
    request_meta: dict,
) -> str | None:
    """If mean confidence is below threshold AND queuing is enabled, persist
    the request for later teacher re-labelling.

    Returns the assigned queue UUID if persisted, else None.
    """
    if not _ENABLED or not lines:
        return None
    confs = [l.confidence for l in lines if l.confidence is not None]
    if not confs:
        return None
    mean_conf = sum(confs) / len(confs)
    # Capture if EITHER low mean-conf (legacy) OR a conf-blind random sample
    # hits (robust to confidence miscalibration). Skip otherwise.
    low_conf = mean_conf < _THRESHOLD
    sampled = (_SAMPLE_RATE > 0.0) and (random.random() < _SAMPLE_RATE)
    if not (low_conf or sampled):
        return None
    capture_reason = "low_conf" if low_conf else "sample"

    with _LOCK:
        if _queue_size_mb() > _MAX_QUEUE_MB:
            # Cap — if the queue grew past budget, stop saving new ones.
            # (Could also FIFO-evict; keep it simple for MVP.)
            return None
        _QUEUE_ROOT.mkdir(parents=True, exist_ok=True)
        (_QUEUE_ROOT / "images").mkdir(exist_ok=True)
        uid = uuid.uuid4().hex[:12]
        img_path = _QUEUE_ROOT / "images" / f"{uid}.jpg"
        img.convert("RGB").save(img_path, "JPEG", quality=92)
        entry = {
            "uid": uid,
            "ts": datetime.now(timezone.utc).isoformat(),
            "image_path": img_path.as_posix(),
            "mean_confidence": round(mean_conf, 4),
            "threshold": _THRESHOLD,
            "capture_reason": capture_reason,
            "n_lines": len(lines),
            "lines": [
                {"text": l.text, "confidence": l.confidence,
                 "bbox": [list(p) for p in l.bbox]}
                for l in lines
            ],
            "request_meta": request_meta,
        }
        with (_QUEUE_ROOT / "queue.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return uid
