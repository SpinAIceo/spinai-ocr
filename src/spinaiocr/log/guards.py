"""Numeric & GPU guards."""
from __future__ import annotations

import logging
import math
from contextlib import contextmanager

log = logging.getLogger(__name__)


def format_tensor_brief(t) -> str:
    """One-line tensor summary safe for log messages."""
    try:
        import torch  # type: ignore
    except ImportError:
        return repr(t)
    if not isinstance(t, torch.Tensor):
        return repr(t)
    if t.numel() == 0:
        return f"Tensor(empty, shape={tuple(t.shape)}, dtype={t.dtype})"
    f = t.detach().float()
    return (
        f"Tensor(shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}, "
        f"min={f.min().item():.3g}, max={f.max().item():.3g}, mean={f.mean().item():.3g}, "
        f"nan={int(torch.isnan(f).sum().item())}, inf={int(torch.isinf(f).sum().item())})"
    )


def check_finite(name: str, value, logger: logging.Logger | None = None, raise_on_bad: bool = True) -> bool:
    """Return True if finite; log ERROR otherwise.

    Accepts Python floats and torch tensors.
    """
    lg = logger or log
    try:
        import torch  # type: ignore

        if hasattr(value, "detach") and hasattr(value, "isnan"):
            nan_count = int(torch.isnan(value).sum().item())
            inf_count = int(torch.isinf(value).sum().item())
            ok = nan_count == 0 and inf_count == 0
        else:
            v = float(value)
            ok = math.isfinite(v)
            nan_count = int(math.isnan(v))
            inf_count = int(math.isinf(v))
    except Exception as e:  # noqa: BLE001
        lg.error("check_finite failed for %s: %s", name, e)
        return False

    if not ok:
        lg.error(
            "non-finite value in %s nan=%d inf=%d detail=%s",
            name, nan_count, inf_count, format_tensor_brief(value),
            extra={"guard": "check_finite", "target": name, "nan": nan_count, "inf": inf_count},
        )
        if raise_on_bad:
            raise FloatingPointError(f"non-finite value in {name} (nan={nan_count}, inf={inf_count})")
    return ok


class NaNInfGuard:
    """Stateful guard: tripped after `max_bad` non-finite values; prints
    a diagnostic report before raising. Wrap the training step body."""

    def __init__(self, max_bad: int = 3, logger: logging.Logger | None = None) -> None:
        self.max_bad = max_bad
        self._bad = 0
        self._logger = logger or log

    def check(self, name: str, value) -> bool:
        try:
            ok = check_finite(name, value, logger=self._logger, raise_on_bad=False)
        except FloatingPointError:
            ok = False
        if not ok:
            self._bad += 1
            if self._bad >= self.max_bad:
                self._logger.critical(
                    "NaNInfGuard tripped after %d bad values in %s — stopping",
                    self._bad, name,
                    extra={"guard": "NaNInfGuard", "bad_count": self._bad},
                )
                raise FloatingPointError(
                    f"NaNInfGuard tripped: {self._bad} non-finite values seen"
                )
        else:
            # reset streak only if we had just 1 — don't mask multi-step instability
            if self._bad == 1:
                self._bad = 0
        return ok


def oom_hint(exc: BaseException, *, batch_size: int | None = None, image_size: int | None = None) -> str:
    """Produce a human-readable hint when we catch a CUDA OOM."""
    lines = ["CUDA out-of-memory. Concrete things to try:"]
    if batch_size is not None:
        lines.append(f"  - halve batch_size ({batch_size} → {max(batch_size // 2, 1)})")
    if image_size is not None:
        lines.append(f"  - shrink input ({image_size} → {max(image_size - 128, 256)})")
    lines += [
        "  - enable bf16 autocast if not already",
        "  - torch.cuda.empty_cache(); del unused tensors before forward",
        "  - use gradient checkpointing on the detection backbone",
        "  - stop other GPU processes (nvidia-smi)",
    ]
    return "\n".join(lines)


@contextmanager
def catch_oom(batch_size: int | None = None, image_size: int | None = None, logger: logging.Logger | None = None):
    """Context manager: convert CUDA OOM into a log+hint+reraise."""
    lg = logger or log
    try:
        yield
    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg or "cuda" in msg and "memory" in msg:
            lg.error(
                "CUDA OOM caught. %s",
                oom_hint(e, batch_size=batch_size, image_size=image_size),
                exc_info=True,
                extra={"guard": "catch_oom", "batch_size": batch_size, "image_size": image_size},
            )
            try:
                import torch  # type: ignore
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
        raise
