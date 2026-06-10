"""SPINAI OCR structured logging.

Design goals:
    * **Find errors fast.** Every error includes:
        - module + function + line
        - full traceback
        - relevant state dump (input shapes, sample index, config key/value)
        - pointer to a per-crash JSON file in `logs/crashes/`
    * **Machine-readable AND human-readable.** Console gets colored Rich
      output; file gets JSONL for grepping / analysis.
    * **Per-module levels.** Noisy modules can be quieted via env vars
      without touching code: `SPINAI_LOG_teachers.paddle=WARNING`.

Usage:
    from spinai_ocr.log import get_logger, setup_logging, capture_crashes
    setup_logging()  # idempotent — safe to call from every entry point
    log = get_logger(__name__)
    log.info("training started", extra={"iters": 5000, "batch_size": 256})

    with capture_crashes("training_loop", extra={"step": step}):
        loss = train_step(...)

Levels:
    TRACE  (5)  — per-tensor-op detail, disabled by default
    DEBUG  (10) — per-batch values
    INFO   (20) — per-iteration summaries, stage timings
    WARNING(30) — skipped sample, fallback used, deprecation
    ERROR  (40) — recoverable failure (sample dropped, one teacher crashed)
    CRITICAL(50)— unrecoverable (training stops)

Environment variables:
    SPINAI_LOG_LEVEL        global min level (default INFO; use DEBUG for dev)
    SPINAI_LOG_DIR          log directory (default `logs/`)
    SPINAI_LOG_JSON=0|1     disable/enable JSONL file handler (default 1)
    SPINAI_LOG_RICH=0|1     disable/enable colored console (default 1)
    SPINAI_LOG_<dotted.mod> override level for a specific module
"""
from __future__ import annotations

from spinai_ocr.log.config import LogConfig, setup_logging
from spinai_ocr.log.context import capture_crashes, log_span, StepLogger
from spinai_ocr.log.guards import (
    NaNInfGuard,
    check_finite,
    format_tensor_brief,
    oom_hint,
)

import logging

_TRACE = 5
logging.addLevelName(_TRACE, "TRACE")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


__all__ = [
    "get_logger",
    "setup_logging",
    "capture_crashes",
    "log_span",
    "StepLogger",
    "LogConfig",
    "NaNInfGuard",
    "check_finite",
    "format_tensor_brief",
    "oom_hint",
]
