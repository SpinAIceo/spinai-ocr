"""Local JSON experiment tracker (+ optional W&B sink).

Layout::

    experiments/
        <project>/
            <run_id>/
                config.json        # hparams at run start
                metrics.jsonl      # one line per log_metrics call
                artifacts.jsonl    # registered artifact paths + meta
                summary.json       # final metrics + status + duration
                README.md          # human-readable summary

`run_id` = `<YYYYMMDD_HHMMSS>_<tag>_<short_sha>`. `tag` is user-supplied,
`short_sha` is the current git commit (best-effort).
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from spinaiocr.io import save_json, save_jsonl_append
from spinaiocr.log import get_logger

log = get_logger("spinaiocr.experiments")


def _short_git_sha() -> str:
    with suppress(Exception):
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
        if out:
            return out
    return uuid.uuid4().hex[:7]


def _safe_tag(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)[:64]


@dataclass
class ExperimentTracker:
    project: str
    run_id: str
    run_dir: Path
    config: dict
    wandb: Any = None
    _started: float = field(default_factory=time.time)
    _status: str = "running"
    _last_metrics: dict = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        tag: str,
        config: dict | None = None,
        project: str = "spinai-ocr",
        root: str | Path = "experiments",
        use_wandb: bool | None = None,
    ) -> "ExperimentTracker":
        run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{_safe_tag(tag)}_{_short_git_sha()}"
        run_dir = Path(root) / project / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg = dict(config or {})
        cfg.setdefault("tag", tag)
        cfg.setdefault("env", {
            "python": os.sys.version.split()[0],
            "host": os.environ.get("COMPUTERNAME", os.environ.get("HOSTNAME", "?")),
            "cuda": _detect_cuda(),
        })
        save_json(run_dir / "config.json", cfg)

        wandb_client = None
        if use_wandb is None:
            use_wandb = os.environ.get("SPINAI_WANDB", "") == "1"
        if use_wandb:
            try:
                import wandb  # type: ignore
                wandb_client = wandb.init(project=project, name=run_id, config=cfg, dir=str(run_dir))
                log.info("wandb.init project=%s run=%s", project, run_id,
                         extra={"op": "wandb.init"})
            except Exception as e:  # noqa: BLE001
                log.warning("wandb unavailable — falling back to local only: %s", e)

        log.info(
            "experiment.start project=%s run=%s dir=%s",
            project, run_id, run_dir,
            extra={"op": "experiment.start", "project": project, "run_id": run_id,
                   "run_dir": str(run_dir)},
        )
        return cls(project=project, run_id=run_id, run_dir=run_dir,
                   config=cfg, wandb=wandb_client)

    # ----- logging -------------------------------------------------------

    def log_metrics(self, step: int, **metrics: Any) -> None:
        record = {"step": step, "ts": time.time(), **metrics}
        save_jsonl_append(self.run_dir / "metrics.jsonl", [record])
        self._last_metrics = metrics
        if self.wandb is not None:
            with suppress(Exception):
                self.wandb.log(metrics, step=step)

    def log_artifact(self, path: str | Path, *, kind: str = "checkpoint", **meta: Any) -> None:
        p = Path(path)
        size = p.stat().st_size if p.exists() else 0
        record = {"path": str(p), "kind": kind, "size_bytes": size, "ts": time.time(), **meta}
        save_jsonl_append(self.run_dir / "artifacts.jsonl", [record])
        log.info("experiment.artifact run=%s kind=%s path=%s size_MB=%.2f",
                 self.run_id, kind, p, size / 1e6,
                 extra={"op": "experiment.artifact", "run_id": self.run_id,
                        "kind": kind, "path": str(p), "size_bytes": size})

    def update_config(self, **kwargs: Any) -> None:
        self.config.update(kwargs)
        save_json(self.run_dir / "config.json", self.config)

    def finish(self, final_metrics: dict | None = None, status: str = "completed") -> None:
        self._status = status
        elapsed = time.time() - self._started
        summary = {
            "run_id": self.run_id, "project": self.project, "status": status,
            "elapsed_s": elapsed, "final_metrics": final_metrics or {},
            "last_metrics": self._last_metrics,
        }
        save_json(self.run_dir / "summary.json", summary)
        readme = (
            f"# {self.run_id}\n\n"
            f"- **project**: {self.project}\n"
            f"- **status**: {status}\n"
            f"- **elapsed**: {elapsed:.1f}s\n"
            f"- **final metrics**: `{json.dumps(final_metrics or {}, ensure_ascii=False)}`\n"
        )
        (self.run_dir / "README.md").write_text(readme, encoding="utf-8")
        if self.wandb is not None:
            with suppress(Exception):
                self.wandb.finish()
        log.info("experiment.done run=%s status=%s elapsed_s=%.1f",
                 self.run_id, status, elapsed,
                 extra={"op": "experiment.done", "run_id": self.run_id,
                        "status": status, "elapsed_s": elapsed})

    # ----- convenience ---------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        status = "failed" if exc_type else "completed"
        self.finish(status=status)


def load_run(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    out = {}
    for name in ("config.json", "summary.json"):
        p = run_dir / name
        if p.exists():
            out[name.replace(".json", "")] = json.loads(p.read_text(encoding="utf-8"))
    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        out["metrics"] = [json.loads(l) for l in metrics_path.read_text(encoding="utf-8").splitlines() if l]
    return out


def _detect_cuda() -> dict:
    info = {"available": False}
    with suppress(Exception):
        import torch  # type: ignore
        info["available"] = bool(torch.cuda.is_available())
        if info["available"]:
            info["device_count"] = torch.cuda.device_count()
            info["name"] = torch.cuda.get_device_name(0)
            info["torch"] = torch.__version__
    return info
