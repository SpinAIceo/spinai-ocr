"""Unified dataset interfaces.

- RecognitionDataset: image + transcription.
- DetectionDataset: image + list of polygons + per-polygon transcription.

Label file format (simple TSV) for recognition:
    image_path\tTEXT
Label file format for detection (one JSON per line, `image_path` relative to root):
    {"image": "x.jpg", "words": [{"points": [[x,y], ...], "text": "..."}]}
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from spinaiocr.log import get_logger
from spinaiocr.vocab.base import Vocab

log = get_logger("spinaiocr.data.dataset")


@dataclass
class RecognitionSample:
    image: np.ndarray
    text: str


class RecognitionDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        label_file: str | Path,
        vocab: Vocab,
        image_height: int = 48,
        max_width: int = 640,
    ) -> None:
        self.root = Path(root)
        self.vocab = vocab
        self.image_height = image_height
        self.max_width = max_width
        self.items: list[tuple[str, str]] = []
        skipped = 0
        with Path(label_file).open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.rstrip("\n")
                if not line:
                    continue
                if "\t" not in line:
                    log.warning("dataset.recognition skip line=%d reason=no_tab file=%s",
                                line_no, str(label_file),
                                extra={"line_no": line_no, "reason": "no_tab"})
                    skipped += 1
                    continue
                path, text = line.split("\t", 1)
                if not text.strip():
                    log.warning("dataset.recognition skip line=%d reason=empty_text path=%s",
                                line_no, path,
                                extra={"line_no": line_no, "reason": "empty_text"})
                    skipped += 1
                    continue
                self.items.append((path, text))
        log.info("dataset.recognition loaded items=%d skipped=%d file=%s",
                 len(self.items), skipped, str(label_file),
                 extra={"loaded": len(self.items), "skipped": skipped})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> RecognitionSample:
        path, text = self.items[idx]
        img = Image.open(self.root / path).convert("RGB")
        w, h = img.size
        arr = np.asarray(
            _msr_resize(img, self.image_height, self.max_width),
            dtype=np.float32,
        ) / 255.0
        return RecognitionSample(image=arr, text=text)


def _msr_resize(img: Image.Image, image_height: int, max_width: int) -> Image.Image:
    """Multi-Size Resizing (SVTRv2): reduce target height for wide-AR crops.

    Wide-aspect crops (dialogue subtitles, signboards) are horizontally
    compressed when forced to a fixed height.  Reducing height for AR>8
    keeps more horizontal character density within the max_width budget.

    AR buckets and target heights (relative to base image_height=48):
      AR ≤ 4:  target_h = image_height       (normal)
      AR 4–8:  target_h = image_height × 0.75 (slight reduction)
      AR > 8:  target_h = image_height × 0.5  (wide subtitle)
    Minimum target_h = 24 to avoid blur artefacts.
    """
    w, h = img.size
    ar = w / max(h, 1)
    if ar > 8:
        target_h = max(image_height // 2, 24)
    elif ar > 4:
        target_h = max(int(image_height * 0.75), 32)
    else:
        target_h = image_height
    new_w = min(int(w * target_h / max(h, 1)), max_width)
    return img.resize((max(new_w, 8), target_h), Image.BILINEAR)


@dataclass
class DetectionSample:
    image: np.ndarray
    polygons: list[np.ndarray]
    texts: list[str]


class DetectionDataset(Dataset):
    def __init__(self, root: str | Path, label_file: str | Path) -> None:
        self.root = Path(root)
        self.items: list[dict] = []
        skipped = 0
        with Path(label_file).open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    log.warning("dataset.detection skip line=%d reason=bad_json err=%s",
                                line_no, e,
                                extra={"line_no": line_no, "reason": "bad_json"})
                    skipped += 1
                    continue
                if not rec.get("image"):
                    log.warning("dataset.detection skip line=%d reason=no_image_key",
                                line_no, extra={"line_no": line_no, "reason": "no_image_key"})
                    skipped += 1
                    continue
                self.items.append(rec)
        log.info("dataset.detection loaded items=%d skipped=%d file=%s",
                 len(self.items), skipped, str(label_file),
                 extra={"loaded": len(self.items), "skipped": skipped})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> DetectionSample:
        item = self.items[idx]
        img = Image.open(self.root / item["image"]).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        polys = [np.array(w["points"], dtype=np.float32) for w in item.get("words", [])]
        texts = [w.get("text", "") for w in item.get("words", [])]
        return DetectionSample(image=arr, polygons=polys, texts=texts)
