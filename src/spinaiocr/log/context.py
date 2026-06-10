"""Context managers for error capture, spans, and per-step summaries."""
from __future__ import annotations

import json
import logging
import socket
import sys
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Crash dumps
# ---------------------------------------------------------------------------


def _crashes_dir() -> Path:
    import os

    return Path(os.environ.get("SPINAI_LOG_DIR", "logs")) / "crashes"


def write_crash_dump(
    tag: str,
    error: BaseException | None,
    extra: dict | None = None,
    traceback_exc_info=None,
) -> Path:
    """Write a self-contained per-crash JSON file.

    Contents:
        - unique id, timestamp, host, pid
        - tag (human-readable scope name e.g. 'training_step')
        - exception type/message/full traceback
        - any 'extra' state supplied by the caller (tensor shapes, step,
          sample index, config key, etc.)

    Returns the path so the log message can point at it.
    """
    crashes = _crashes_dir()
    crashes.mkdir(parents=True, exist_ok=True)
    cid = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    path = crashes / f"{tag}_{cid}.json"

    exc_info = traceback_exc_info
    if exc_info is None and error is not None:
        exc_info = (type(error), error, error.__traceback__)
    tb_str = ""
    exc_type = ""
    exc_msg = ""
    if exc_info and exc_info[0] is not None:
        tb_str = "".join(traceback.format_exception(*exc_info))
        exc_type = exc_info[0].__name__
        exc_msg = str(exc_info[1])

    payload = {
        "id": cid,
        "timestamp": time.time(),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": socket.gethostname(),
        "tag": tag,
        "exception": {
            "type": exc_type,
            "message": exc_msg,
            "traceback": tb_str,
        },
        "extra": _safe(extra or {}),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


@contextmanager
def capture_crashes(tag: str, extra: dict | None = None, reraise: bool = True, logger_name: str | None = None):
    """Wrap risky code blocks. On exception:
        - log ERROR with full traceback + tag
        - write a crash dump to `logs/crashes/<tag>_<id>.json`
        - re-raise (unless `reraise=False`)
    """
    lg = logging.getLogger(logger_name or f"spinaiocr.capture.{tag}")
    started = time.perf_counter()
    try:
        yield
    except BaseException as e:
        dump_path = write_crash_dump(tag=tag, error=e, extra=extra)
        lg.error(
            "crash in %s: %s: %s  dump=%s",
            tag, type(e).__name__, e, dump_path,
            exc_info=True,
            extra={"tag": tag, "dump_path": str(dump_path), **(extra or {})},
        )
        if reraise:
            raise
    finally:
        elapsed_ms = (time.perf_counter() - started) * 1000
        lg.debug("span.end tag=%s elapsed_ms=%.1f", tag, elapsed_ms,
                 extra={"tag": tag, "elapsed_ms": elapsed_ms})


# ---------------------------------------------------------------------------
# Timed spans
# ---------------------------------------------------------------------------


@contextmanager
def log_span(name: str, logger_name: str | None = None, level: int = logging.INFO, **extra):
    """Log start/end with elapsed time. For stage profiling in pipelines."""
    lg = logging.getLogger(logger_name or "spinaiocr.span")
    start = time.perf_counter()
    lg.log(level, "span.start name=%s", name, extra={"span": name, "event": "start", **extra})
    try:
        yield
    except BaseException:
        elapsed_ms = (time.perf_counter() - start) * 1000
        lg.error(
            "span.fail name=%s elapsed_ms=%.1f",
            name, elapsed_ms,
            exc_info=True,
            extra={"span": name, "event": "fail", "elapsed_ms": elapsed_ms, **extra},
        )
        raise
    else:
        elapsed_ms = (time.perf_counter() - start) * 1000
        lg.log(
            level,
            "span.end name=%s elapsed_ms=%.1f",
            name, elapsed_ms,
            extra={"span": name, "event": "end", "elapsed_ms": elapsed_ms, **extra},
        )


# ---------------------------------------------------------------------------
# Step-level training log aggregator
# ---------------------------------------------------------------------------


@dataclass
class StepLogger:
    """Buffer per-step metrics and log summaries at configurable cadence.

    Example::
        sl = StepLogger("train.reco", log_every=100)
        for step in range(iters):
            loss = ...
            sl.push(step, loss=loss, lr=lr, grad_norm=gn)
        sl.close()
    """

    name: str
    log_every: int = 100
    window: int = 20
    _logger: logging.Logger = field(init=False)
    _buf: list[dict] = field(default_factory=list, init=False)
    _started: float = field(default_factory=time.perf_counter, init=False)
    _last_step_time: float = field(default_factory=time.perf_counter, init=False)

    def __post_init__(self) -> None:
        self._logger = logging.getLogger(f"spinaiocr.step.{self.name}")

    def push(self, step: int, **metrics: Any) -> None:
        now = time.perf_counter()
        dt = now - self._last_step_time
        self._last_step_time = now
        entry = {"step": step, "dt_s": dt, **metrics}
        self._buf.append(entry)
        if step % self.log_every == 0 or step == 0:
            self._emit_summary(step)

    def _emit_summary(self, step: int) -> None:
        recent = self._buf[-self.window:]
        if not recent:
            return
        agg = {}
        for key in recent[0]:
            if key == "step":
                continue
            vals = [e[key] for e in recent if isinstance(e[key], (int, float))]
            if not vals:
                continue
            agg[key] = sum(vals) / len(vals)
        it_per_s = 1.0 / agg["dt_s"] if "dt_s" in agg and agg["dt_s"] > 0 else 0.0
        self._logger.info(
            "step=%d  %s  it/s=%.1f",
            step,
            "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                      for k, v in agg.items() if k != "dt_s"),
            it_per_s,
            extra={"step": step, "window": len(recent), "it_per_s": it_per_s, **agg},
        )

    def close(self, final_metrics: dict | None = None) -> None:
        total = time.perf_counter() - self._started
        steps = len(self._buf)
        payload = {"total_steps": steps, "total_s": total, "it_per_s": steps / max(total, 1e-6)}
        if final_metrics:
            payload.update(final_metrics)
        self._logger.info("training complete %s", _fmt(payload), extra=payload)


def _fmt(d: dict) -> str:
    parts = []
    for k, v in d.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
        else:
            parts.append(f"{k}={v}")
    return "  ".join(parts)


def _safe(obj):
    """Best-effort JSON coercion."""
    if isinstance(obj, dict):
        return {str(k): _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_safe(x) for x in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # common torch types
    try:
        import torch  # type: ignore

        if isinstance(obj, torch.Tensor):
            return {
                "__tensor__": True,
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "device": str(obj.device),
                "mean": float(obj.float().mean().item()) if obj.numel() else None,
                "min": float(obj.float().min().item()) if obj.numel() else None,
                "max": float(obj.float().max().item()) if obj.numel() else None,
            }
    except Exception:  # noqa: BLE001
        pass
    return str(obj)
