"""Model registry — track every checkpoint with its metrics + metadata.

    registry root (default: `checkpoints/registry/`)
        index.jsonl                 append-only log of registered models
        <kind>/<lang>/<tag>/<ver>/  actual ckpt + metrics + config

`kind` = detection | recognition | angle | refiner
`lang` = ko | en | ja | zh | multi | _
`tag`  = free-form (e.g. "svtr_lite_v2", "dbnet_r18")
`ver`  = auto-increment per (kind, lang, tag)

Usage::

    reg = ModelRegistry()
    reg.register(
        kind="recognition", lang="ko", tag="svtr_lite",
        ckpt_path="checkpoints/subset_ko/rec.pth",
        metrics={"cer": 0.1128, "wer": 0.45},
        config={"arch": "svtr_lite", "vocab_size": 131},
    )
    best = reg.best("recognition", lang="ko", metric="cer")
    print(best.ckpt_path, best.metrics["cer"])
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from spinaiocr.io import save_json, save_jsonl_append, sha256_of_file
from spinaiocr.log import get_logger

log = get_logger("spinaiocr.models.registry")


@dataclass
class ModelEntry:
    kind: str
    lang: str
    tag: str
    version: int
    registered_at: float
    ckpt_path: str
    size_bytes: int
    sha256: str
    metrics: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    notes: str = ""

    @property
    def run_dir(self) -> Path:
        return Path(self.ckpt_path).parent


class ModelRegistry:
    def __init__(self, root: str | Path = "checkpoints/registry") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._index = self.root / "index.jsonl"

    # ----- write ---------------------------------------------------------

    def register(
        self,
        *,
        kind: str,
        lang: str,
        tag: str,
        ckpt_path: str | Path,
        metrics: dict | None = None,
        config: dict | None = None,
        notes: str = "",
    ) -> ModelEntry:
        ckpt_src = Path(ckpt_path)
        if not ckpt_src.exists():
            raise FileNotFoundError(ckpt_src)

        existing = [e for e in self.list_entries()
                    if e.kind == kind and e.lang == lang and e.tag == tag]
        version = max((e.version for e in existing), default=0) + 1

        target_dir = self.root / kind / lang / tag / f"v{version:03d}"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_ckpt = target_dir / ckpt_src.name
        shutil.copy2(ckpt_src, target_ckpt)

        entry = ModelEntry(
            kind=kind, lang=lang, tag=tag, version=version,
            registered_at=time.time(),
            ckpt_path=str(target_ckpt),
            size_bytes=target_ckpt.stat().st_size,
            sha256=sha256_of_file(target_ckpt),
            metrics=dict(metrics or {}),
            config=dict(config or {}),
            notes=notes,
        )
        save_json(target_dir / "entry.json", asdict(entry))
        save_jsonl_append(self._index, [asdict(entry)])
        log.info(
            "registry.registered kind=%s lang=%s tag=%s v=%d ckpt=%s metrics=%s",
            kind, lang, tag, version, target_ckpt, metrics or {},
            extra={"op": "registry.register", "kind": kind, "lang": lang,
                   "tag": tag, "version": version, "sha256_prefix": entry.sha256[:12]},
        )
        return entry

    # ----- read ----------------------------------------------------------

    def list_entries(
        self,
        kind: str | None = None,
        lang: str | None = None,
        tag: str | None = None,
    ) -> list[ModelEntry]:
        if not self._index.exists():
            return []
        rows: list[ModelEntry] = []
        for line in self._index.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if kind and d.get("kind") != kind:
                continue
            if lang and d.get("lang") != lang:
                continue
            if tag and d.get("tag") != tag:
                continue
            rows.append(ModelEntry(**d))
        return rows

    def best(
        self,
        kind: str,
        *,
        lang: str | None = None,
        tag: str | None = None,
        metric: str = "cer",
        lower_is_better: bool = True,
    ) -> ModelEntry | None:
        candidates = [
            e for e in self.list_entries(kind, lang, tag)
            if metric in e.metrics
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda e: e.metrics[metric], reverse=not lower_is_better)
        return candidates[0]

    def promote_to_current(
        self,
        entry: ModelEntry,
        checkpoints_root: str | Path = "checkpoints",
        tier: str | None = None,
    ) -> Path:
        """Copy the entry's ckpt to the canonical `checkpoints/<tier>/<lang>/<name>.pth`
        path the inference pipeline expects. Returns the destination.

        iter 140: `tier` is now an explicit parameter. Pre-fix the tier was
        hardcoded to "lite", which silently overwrote the wrong checkpoint
        when promoting a consumer_v1 / medical entry (the production tiers
        added after iter 50). When `tier` is None we look at `entry.config`
        for a "tier" key, then fall back to "lite" for backwards-compat with
        entries registered before this fix.
        """
        if tier is None:
            tier = entry.config.get("tier", "lite") if entry.config else "lite"
        dest_dir = Path(checkpoints_root) / tier / entry.lang
        dest_dir.mkdir(parents=True, exist_ok=True)
        name_by_kind = {"detection": "det", "recognition": "rec",
                        "angle": "angle", "refiner": "refiner"}
        dest = dest_dir / f"{name_by_kind.get(entry.kind, entry.kind)}.pth"
        shutil.copy2(entry.ckpt_path, dest)
        log.info("registry.promoted src=%s tier=%s dest=%s",
                 entry.ckpt_path, tier, dest,
                 extra={"op": "registry.promote", "entry": asdict(entry),
                        "tier": tier, "dest": str(dest)})
        return dest
