"""Logging setup entry point."""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from spinai_ocr.log.formatters import ConsoleFormatter, JsonLinesFormatter


_CONFIGURED = False


@dataclass
class LogConfig:
    level: str = "INFO"
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    enable_json: bool = True
    enable_rich: bool = True
    max_bytes: int = 16 * 1024 * 1024  # 16 MB
    backup_count: int = 8
    per_module_levels: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "LogConfig":
        """Read env vars at call time.

        Per-module overrides are supplied via ``SPINAI_LOG_LEVELS`` as JSON:
            SPINAI_LOG_LEVELS='{"spinai_ocr.teachers": "DEBUG"}'
        """
        cfg = cls(
            level=os.environ.get("SPINAI_LOG_LEVEL", "INFO"),
            log_dir=Path(os.environ.get("SPINAI_LOG_DIR", "logs")),
            enable_json=os.environ.get("SPINAI_LOG_JSON", "1") != "0",
            enable_rich=os.environ.get("SPINAI_LOG_RICH", "1") != "0",
        )
        raw = os.environ.get("SPINAI_LOG_LEVELS")
        if raw:
            import json as _json
            try:
                cfg.per_module_levels = dict(_json.loads(raw))
            except Exception:  # noqa: BLE001
                pass
        return cfg


def setup_logging(config: LogConfig | None = None, force: bool = False) -> LogConfig:
    """Idempotent. Safe to call from every CLI entry point."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return _existing_config()

    cfg = config or LogConfig.from_env()
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    (cfg.log_dir / "crashes").mkdir(exist_ok=True)

    root = logging.getLogger()
    root.setLevel(getattr(logging, cfg.level.upper(), logging.INFO))
    # clear any prior handlers (e.g. from third-party calls to basicConfig)
    for h in list(root.handlers):
        root.removeHandler(h)

    # --- console ---
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(ConsoleFormatter(use_color=cfg.enable_rich))
    console.setLevel(root.level)
    root.addHandler(console)

    # --- rotating text file (every line readable) ---
    text_path = cfg.log_dir / "spinai.log"
    text_handler = logging.handlers.RotatingFileHandler(
        text_path,
        maxBytes=cfg.max_bytes,
        backupCount=cfg.backup_count,
        encoding="utf-8",
    )
    text_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s:%(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    text_handler.setLevel(logging.DEBUG)  # file captures more than console
    root.addHandler(text_handler)

    # --- JSONL structured (all levels) ---
    if cfg.enable_json:
        jsonl_path = cfg.log_dir / "spinai.jsonl"
        json_handler = logging.handlers.RotatingFileHandler(
            jsonl_path,
            maxBytes=cfg.max_bytes,
            backupCount=cfg.backup_count,
            encoding="utf-8",
        )
        json_handler.setFormatter(JsonLinesFormatter())
        json_handler.setLevel(logging.DEBUG)
        root.addHandler(json_handler)

        # --- errors.jsonl: WARNING+ only, for batch error review ---
        errors_path = cfg.log_dir / "errors.jsonl"
        err_handler = logging.handlers.RotatingFileHandler(
            errors_path,
            maxBytes=cfg.max_bytes // 4,  # smaller — only warnings/errors
            backupCount=4,
            encoding="utf-8",
        )
        err_handler.setFormatter(JsonLinesFormatter())
        err_handler.setLevel(logging.WARNING)
        root.addHandler(err_handler)

    # --- per-module overrides ---
    for mod, lvl in cfg.per_module_levels.items():
        logging.getLogger(mod).setLevel(getattr(logging, lvl.upper(), logging.INFO))

    # Tame noisy libraries unless explicitly overridden
    for noisy in ("PIL", "urllib3", "matplotlib", "huggingface_hub", "filelock"):
        if noisy not in cfg.per_module_levels:
            logging.getLogger(noisy).setLevel(logging.WARNING)

    # Capture unhandled exceptions
    _install_excepthook(cfg)

    _CONFIGURED = True
    logging.getLogger(__name__).debug(
        "logging configured level=%s dir=%s json=%s rich=%s",
        cfg.level, cfg.log_dir, cfg.enable_json, cfg.enable_rich,
    )
    return cfg


_CURRENT: LogConfig | None = None


def _existing_config() -> LogConfig:
    assert _CURRENT is not None
    return _CURRENT


def _install_excepthook(cfg: LogConfig) -> None:
    previous = sys.excepthook

    def _hook(exc_type, exc, tb):
        logging.getLogger("spinai_ocr.unhandled").critical(
            "unhandled exception", exc_info=(exc_type, exc, tb)
        )
        from spinai_ocr.log.context import write_crash_dump

        write_crash_dump(
            tag="unhandled",
            error=exc,
            extra={},
            traceback_exc_info=(exc_type, exc, tb),
        )
        previous(exc_type, exc, tb)

    sys.excepthook = _hook
