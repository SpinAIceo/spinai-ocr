"""Dataset sanity checks.

Catches common label/data problems before they corrupt training:
    * Missing or unreadable image files
    * Invalid or degenerate polygons (area ≤ 0, colinear, self-intersecting)
    * Polygon fully outside image bounds
    * Empty or suspiciously short text
    * Text characters not in the vocab
    * Duplicate images (content-hash collision)
    * Tiny crops (height < threshold) that hurt recognition training
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon

from spinaiocr.vocab.base import Vocab, load_vocab


@dataclass
class ValidationReport:
    n_records: int = 0
    missing_images: list[str] = field(default_factory=list)
    invalid_polygons: list[tuple[str, int]] = field(default_factory=list)
    empty_text: list[tuple[str, int]] = field(default_factory=list)
    oov_chars: Counter = field(default_factory=Counter)
    tiny_crops: list[tuple[str, int]] = field(default_factory=list)
    duplicate_hashes: list[list[str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_records": self.n_records,
            "missing_images": len(self.missing_images),
            "invalid_polygons": len(self.invalid_polygons),
            "empty_text": len(self.empty_text),
            "tiny_crops": len(self.tiny_crops),
            "duplicate_groups": len(self.duplicate_hashes),
            "oov_top20": self.oov_chars.most_common(20),
        }

    def is_clean(self) -> bool:
        return (
            not self.missing_images
            and not self.invalid_polygons
            and not self.empty_text
            and not self.duplicate_hashes
        )


def validate_detection_jsonl(
    jsonl_path: Path,
    images_root: Path,
    vocab: Vocab | str | None = None,
    min_crop_height: int = 8,
) -> ValidationReport:
    vocab_obj = load_vocab(vocab) if isinstance(vocab, str) else vocab
    report = ValidationReport()
    hashes: dict[str, list[str]] = {}

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            report.n_records += 1
            img_rel = rec.get("image", "")
            img_path = images_root / img_rel
            if not img_path.exists():
                report.missing_images.append(img_rel)
                continue

            raw = img_path.read_bytes()
            h = hashlib.sha1(raw).hexdigest()
            hashes.setdefault(h, []).append(img_rel)

            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                report.missing_images.append(img_rel)
                continue
            H, W = img.shape[:2]

            for i, w in enumerate(rec.get("words", [])):
                pts = w.get("points")
                text = w.get("text", "")
                if not pts or len(pts) < 3:
                    report.invalid_polygons.append((img_rel, i))
                    continue
                poly = np.array(pts, dtype=np.float32)
                sh = Polygon(poly)
                if not sh.is_valid or sh.area <= 1:
                    report.invalid_polygons.append((img_rel, i))
                    continue
                if (
                    poly[:, 0].max() < 0
                    or poly[:, 0].min() > W
                    or poly[:, 1].max() < 0
                    or poly[:, 1].min() > H
                ):
                    report.invalid_polygons.append((img_rel, i))
                    continue

                if not text or text == "###":
                    continue  # legitimate ignore marker
                if len(text.strip()) == 0:
                    report.empty_text.append((img_rel, i))

                heights = np.linalg.norm(poly[0] - poly[3])
                if heights < min_crop_height:
                    report.tiny_crops.append((img_rel, i))

                if vocab_obj is not None:
                    for ch in text:
                        if ch not in vocab_obj._ctoi:  # noqa: SLF001
                            report.oov_chars[ch] += 1

    for h, paths in hashes.items():
        if len(paths) > 1:
            report.duplicate_hashes.append(paths)

    return report


def validate_recognition_tsv(
    tsv_path: Path,
    images_root: Path,
    vocab: Vocab | str | None = None,
    min_height: int = 8,
) -> ValidationReport:
    vocab_obj = load_vocab(vocab) if isinstance(vocab, str) else vocab
    report = ValidationReport()
    hashes: dict[str, list[str]] = {}

    with tsv_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.rstrip("\n")
            if "\t" not in line:
                continue
            rel, text = line.split("\t", 1)
            report.n_records += 1
            path = images_root / rel
            if not path.exists():
                report.missing_images.append(rel)
                continue
            raw = path.read_bytes()
            h = hashlib.sha1(raw).hexdigest()
            hashes.setdefault(h, []).append(rel)

            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                report.missing_images.append(rel)
                continue
            if img.shape[0] < min_height:
                report.tiny_crops.append((rel, idx))
            if not text.strip():
                report.empty_text.append((rel, idx))
            if vocab_obj is not None:
                for ch in text:
                    if ch not in vocab_obj._ctoi:  # noqa: SLF001
                        report.oov_chars[ch] += 1

    for h, paths in hashes.items():
        if len(paths) > 1:
            report.duplicate_hashes.append(paths)

    return report
