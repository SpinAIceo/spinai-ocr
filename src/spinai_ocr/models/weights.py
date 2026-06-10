"""Weight download helpers.

External model weights are not redistributed by SPINAI OCR — each model has
its own license, and many require manual acceptance. This module only
resolves paths and (when a URL is available) downloads into
``checkpoints/<name>/``.

For now we keep a tiny registry with URLs the user can override via env var:

    SPINAI_DEXINED_URL=https://... pip install ... && spinai-ocr ...
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

CHECKPOINT_ROOT = Path(os.environ.get("SPINAI_CKPT_ROOT", "checkpoints"))


@dataclass
class WeightEntry:
    name: str
    filename: str
    default_url: str | None
    sha256: str | None
    license: str
    note: str = ""

    def env_var(self) -> str:
        return f"SPINAI_{self.name.upper()}_URL"


REGISTRY: dict[str, WeightEntry] = {
    "dexined": WeightEntry(
        name="dexined",
        filename="dexined_10_model.pth",
        default_url=None,  # user must provide via env or manual download
        sha256=None,
        license="MIT",
        note="Download from https://github.com/xavysp/DexiNed (official release assets).",
    ),
    "trocr_printed": WeightEntry(
        name="trocr_printed",
        filename="trocr_base_printed",
        default_url=None,
        sha256=None,
        license="MIT",
        note="HuggingFace: microsoft/trocr-base-printed — loaded via transformers, not this helper.",
    ),
    "sam_vit_b": WeightEntry(
        name="sam_vit_b",
        filename="sam_vit_b_01ec64.pth",
        default_url="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        sha256=None,
        license="Apache-2.0",
        note="Meta SAM ViT-B base model.",
    ),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(name: str, download: bool = True) -> Path:
    if name not in REGISTRY:
        raise KeyError(f"Unknown weight entry: {name}")
    entry = REGISTRY[name]
    target = CHECKPOINT_ROOT / entry.name / entry.filename
    if target.exists():
        return target
    url = os.environ.get(entry.env_var()) or entry.default_url
    if not url:
        raise FileNotFoundError(
            f"Weight '{name}' not found at {target}. "
            f"Either set {entry.env_var()} or manually place the file. "
            f"Note: {entry.note}"
        )
    if not download:
        raise FileNotFoundError(f"Weight '{name}' missing and download=False")
    target.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, target)
    if entry.sha256 and _sha256(target) != entry.sha256:
        target.unlink(missing_ok=True)
        raise RuntimeError(f"Checksum mismatch for '{name}'")
    return target
