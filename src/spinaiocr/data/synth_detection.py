"""Synthetic detection-mode pages.

Unlike `synth_simple.py` (single-line recognition crops), this renders full
document-like images with multiple text lines placed at varying positions,
fonts, and sizes. The output matches :class:`DetectionDataset` input:

    out_dir/images/*.png
    out_dir/labels.jsonl   (one {"image": ..., "words": [...]} per line)

Used to warm up DBNet before real labeled data is available.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from spinaiocr.data.synth_simple import _list_fonts, DEFAULT_FONT_DIR


@dataclass
class DetectionSynthSpec:
    out_dir: Path
    corpus: list[str]
    count: int = 1000
    image_size: tuple[int, int] = (640, 640)  # (h, w)
    min_lines: int = 3
    max_lines: int = 12
    font_size_range: tuple[int, int] = (18, 40)
    font_dirs: tuple[Path, ...] = (DEFAULT_FONT_DIR / "ko", DEFAULT_FONT_DIR / "en")
    background_noise: bool = True
    rotation_deg: float = 0.0  # per-line rotation cap


def _rand_bg(h: int, w: int) -> Image.Image:
    base = random.randint(230, 255)
    img = Image.new("RGB", (w, h), (base, base, base))
    if random.random() < 0.4:
        arr = np.asarray(img).astype(np.int16)
        noise = np.random.randint(-8, 8, arr.shape, dtype=np.int16)
        img = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))
    return img


def _overlaps(bbox: tuple[int, int, int, int], placed: list[tuple[int, int, int, int]]) -> bool:
    x0, y0, x1, y1 = bbox
    for px0, py0, px1, py1 in placed:
        if not (x1 <= px0 or x0 >= px1 or y1 <= py0 or y0 >= py1):
            return True
    return False


def _render_line(text: str, font: ImageFont.FreeTypeFont, rotation_deg: float) -> tuple[Image.Image, np.ndarray]:
    """Render a single text line as a transparent RGBA patch + return its
    4-point polygon in patch-local coords (pre-rotation corners of the glyph
    tight box). After rotation we transform the corners with the rotation
    matrix."""
    bbox = font.getbbox(text)
    pad = 2
    tw = bbox[2] - bbox[0] + pad * 2
    th = bbox[3] - bbox[1] + pad * 2
    patch = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    draw = ImageDraw.Draw(patch)
    color = (random.randint(0, 50), random.randint(0, 50), random.randint(0, 50), 255)
    draw.text((pad - bbox[0], pad - bbox[1]), text, fill=color, font=font)
    # polygon in patch-local (no rotation)
    poly = np.array(
        [[0, 0], [tw, 0], [tw, th], [0, th]], dtype=np.float32
    )
    if rotation_deg != 0 and random.random() < 0.3:
        angle = random.uniform(-rotation_deg, rotation_deg)
        patch = patch.rotate(angle, expand=True, fillcolor=(0, 0, 0, 0))
        # rotation expands — recompute polygon via PIL's transform
        # (approximate with new axis-aligned box — adequate for loose DBNet GT)
        nw, nh = patch.size
        poly = np.array([[0, 0], [nw, 0], [nw, nh], [0, nh]], dtype=np.float32)
    return patch, poly


def generate_detection_pages(spec: DetectionSynthSpec) -> int:
    spec.out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = spec.out_dir / "images"
    img_dir.mkdir(exist_ok=True)
    label_path = spec.out_dir / "labels.jsonl"

    fonts = _list_fonts(spec.font_dirs)
    if not fonts:
        raise RuntimeError(
            f"No fonts under {list(spec.font_dirs)}; run scripts/collect_fonts.py first"
        )
    if not spec.corpus:
        raise ValueError("corpus empty")

    h, w = spec.image_size
    written = 0
    with label_path.open("w", encoding="utf-8") as flabels:
        for i in range(spec.count):
            page = _rand_bg(h, w)
            placed_boxes: list[tuple[int, int, int, int]] = []
            words: list[dict] = []

            n_lines = random.randint(spec.min_lines, spec.max_lines)
            tries = 0
            while len(words) < n_lines and tries < n_lines * 10:
                tries += 1
                text = random.choice(spec.corpus)
                try:
                    font = ImageFont.truetype(
                        str(random.choice(fonts)),
                        random.randint(*spec.font_size_range),
                    )
                except Exception:  # noqa: BLE001
                    continue
                try:
                    patch, local_poly = _render_line(text, font, spec.rotation_deg)
                except Exception:  # noqa: BLE001
                    continue
                pw, ph = patch.size
                if pw >= w or ph >= h:
                    continue
                x = random.randint(0, w - pw)
                y = random.randint(0, h - ph)
                bbox = (x, y, x + pw, y + ph)
                if _overlaps(bbox, placed_boxes):
                    continue
                page.paste(patch, (x, y), patch)
                placed_boxes.append(bbox)
                world_poly = (local_poly + np.array([x, y], dtype=np.float32)).tolist()
                words.append({"points": world_poly, "text": text})

            if spec.background_noise and random.random() < 0.2:
                page = page.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8)))

            fn = f"{i:08d}.png"
            page.save(img_dir / fn)
            flabels.write(
                json.dumps({"image": f"images/{fn}", "words": words}, ensure_ascii=False)
                + "\n"
            )
            written += 1
    return written
