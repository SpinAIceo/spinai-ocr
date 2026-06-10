"""Log formatters."""
from __future__ import annotations

import json
import logging
import os
import socket
import sys
import time
import traceback


_COLORS = {
    "TRACE": "\033[38;5;245m",     # dim gray
    "DEBUG": "\033[38;5;109m",     # soft cyan
    "INFO": "\033[38;5;114m",      # green
    "WARNING": "\033[38;5;214m",   # orange
    "ERROR": "\033[38;5;203m",     # red
    "CRITICAL": "\033[48;5;124;97m",  # white on dark red
}
_RESET = "\033[0m"
_DIM = "\033[2m"


def _supports_color() -> bool:
    if sys.platform == "win32":
        # Windows Terminal sets WT_SESSION; ANSICON is the older toggle.
        return any(k in os.environ for k in ("WT_SESSION", "ANSICON", "TERM_PROGRAM"))
    return sys.stderr.isatty()


class ConsoleFormatter(logging.Formatter):
    """Compact, colored, one-line per record.

        HH:MM:SS.mmm  LEVEL  module:line  message  {extra}

    Exceptions are printed on following indented lines.
    """

    def __init__(self, use_color: bool = True) -> None:
        super().__init__()
        self.use_color = use_color and _supports_color()

    def format(self, record: logging.LogRecord) -> str:
        ts = time.strftime("%H:%M:%S", time.localtime(record.created))
        ts = f"{ts}.{int(record.msecs):03d}"
        level = record.levelname
        if self.use_color:
            color = _COLORS.get(level, "")
            level_str = f"{color}{level:<8}{_RESET}"
            loc = f"{_DIM}{record.name}:{record.lineno}{_RESET}"
        else:
            level_str = f"{level:<8}"
            loc = f"{record.name}:{record.lineno}"

        msg = record.getMessage()
        extra = _extract_extra(record)
        extra_str = ""
        if extra:
            # compact key=value formatting
            pairs = [f"{k}={_brief(v)}" for k, v in extra.items()]
            joined = " ".join(pairs)
            extra_str = f"  {_DIM}{joined}{_RESET}" if self.use_color else f"  {joined}"

        out = f"{ts}  {level_str}  {loc}  {msg}{extra_str}"

        if record.exc_info:
            out += "\n" + "".join(traceback.format_exception(*record.exc_info)).rstrip()
        return out


class JsonLinesFormatter(logging.Formatter):
    """One JSON object per record. Safe to `jq` / analyze."""

    _HOST = socket.gethostname()
    _PID = os.getpid()

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": record.created,
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
            "msg": record.getMessage(),
            "host": self._HOST,
            "pid": self._PID,
        }
        extra = _extract_extra(record)
        if extra:
            payload["extra"] = _jsonable(extra)
        if record.exc_info:
            payload["exc"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": "".join(traceback.format_exception(*record.exc_info)),
            }
        return json.dumps(payload, ensure_ascii=False, default=str)


# ---------- helpers -----------------------------------------------------------


_STANDARD_ATTRS = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName",
}


def _extract_extra(record: logging.LogRecord) -> dict:
    """Pull user-supplied `extra` fields off the LogRecord."""
    return {
        k: v
        for k, v in record.__dict__.items()
        if k not in _STANDARD_ATTRS and not k.startswith("_")
    }


def _brief(v) -> str:
    s = repr(v)
    if len(s) > 120:
        s = s[:117] + "..."
    return s


def _jsonable(obj):
    """Coerce arbitrary values to JSON-safe types."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)
