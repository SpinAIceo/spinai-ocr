"""Lightweight synthetic recognition sample generator.

Pure PIL implementation — no `trdg` dependency. Works well for Korean since
we control font selection directly. Produces single-line images suitable for
CTC training.

Output:
    out_dir/labels.tsv   (image\ttext)
    out_dir/00000000.png, 00000001.png, ...
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from spinaiocr.log import get_logger

log = get_logger("spinaiocr.data.synth_simple")


DEFAULT_FONT_DIR = Path(__file__).parent.parent / "assets" / "fonts"


def _list_fonts(font_dirs: Iterable[Path]) -> list[Path]:
    fonts: list[Path] = []
    for d in font_dirs:
        if d.exists():
            fonts.extend([p for p in d.rglob("*") if p.suffix.lower() in {".ttf", ".otf"}])
    return fonts


@dataclass
class SimpleSynthSpec:
    out_dir: Path
    corpus: list[str]  # pre-filtered list of short strings
    count: int = 10_000
    image_height: int = 48
    font_size_range: tuple[int, int] = (28, 42)
    padding: int = 6
    font_dirs: tuple[Path, ...] = (DEFAULT_FONT_DIR / "ko", DEFAULT_FONT_DIR / "en")
    background_jitter: bool = True
    blur_p: float = 0.25
    rotation_deg: float = 2.0


def _random_bg(w: int, h: int) -> Image.Image:
    base = random.randint(210, 255)
    bg = Image.new("RGB", (w, h), (base, base, base))
    if random.random() < 0.3:
        # add subtle gradient noise
        arr = np.asarray(bg).astype(np.int16)
        noise = np.random.randint(-10, 10, arr.shape, dtype=np.int16)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        bg = Image.fromarray(arr)
    return bg


def _random_fg_color() -> tuple[int, int, int]:
    # mostly dark
    v = random.randint(0, 60)
    return (v, v, v)


def _render_one(text: str, font: ImageFont.FreeTypeFont, spec: SimpleSynthSpec) -> Image.Image:
    # measure
    bbox = font.getbbox(text)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    canvas_w = tw + spec.padding * 2
    canvas_h = max(spec.image_height, th + spec.padding * 2)

    img = _random_bg(canvas_w, canvas_h) if spec.background_jitter else Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(img)
    draw.text((spec.padding - bbox[0], (canvas_h - th) // 2 - bbox[1]), text, fill=_random_fg_color(), font=font)

    if spec.rotation_deg > 0 and random.random() < 0.3:
        angle = random.uniform(-spec.rotation_deg, spec.rotation_deg)
        img = img.rotate(angle, expand=False, fillcolor=(255, 255, 255))

    if spec.blur_p > 0 and random.random() < spec.blur_p:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.2)))

    # normalize height
    if img.height != spec.image_height:
        new_w = max(int(img.width * spec.image_height / img.height), 8)
        img = img.resize((new_w, spec.image_height), Image.BILINEAR)
    return img


def generate_simple(spec: SimpleSynthSpec) -> int:
    import time as _time

    spec.out_dir.mkdir(parents=True, exist_ok=True)
    font_paths = _list_fonts(spec.font_dirs)
    if not font_paths:
        log.error("synth.no_fonts dirs=%s", list(spec.font_dirs))
        raise RuntimeError(
            f"No fonts found under {list(spec.font_dirs)}. "
            "Run `python scripts/collect_fonts.py` first."
        )
    if not spec.corpus:
        raise ValueError("corpus is empty")
    # _47: loud warning when corpus is too small. subset_ko was generated
    # from only 21 distinct strings, which baked memorisation into the
    # production recognizer and explains the OOD failures in _44/_46.
    n_unique = len(set(spec.corpus))
    if n_unique < 200:
        log.warning(
            "synth.corpus_tiny n_unique=%d count=%d — model will memorise these "
            "strings. For general-purpose Korean OCR, use >=5000 unique corpus "
            "lines (e.g. data/corpora/ko_wiki_combined.txt has 4822). See _47.",
            n_unique, spec.count,
            extra={"op": "synth_recognition", "n_unique": n_unique,
                   "count": spec.count},
        )

    log.info(
        "synth.begin out=%s count=%d fonts=%d corpus=%d unique=%d h=%d",
        spec.out_dir, spec.count, len(font_paths), len(spec.corpus),
        n_unique, spec.image_height,
        extra={"op": "synth_recognition", "out": str(spec.out_dir),
               "count": spec.count, "n_fonts": len(font_paths),
               "corpus_size": len(spec.corpus)},
    )

    written = 0
    font_errors = 0
    render_errors = 0
    started = _time.perf_counter()
    progress_every = max(1, spec.count // 10)
    with (spec.out_dir / "labels.tsv").open("w", encoding="utf-8") as f:
        for idx in range(spec.count):
            text = random.choice(spec.corpus)
            font_path = random.choice(font_paths)
            try:
                font = ImageFont.truetype(str(font_path), random.randint(*spec.font_size_range))
            except Exception as e:  # noqa: BLE001
                font_errors += 1
                log.debug("synth.font_err path=%s err=%s", font_path, e)
                continue
            try:
                img = _render_one(text, font, spec)
            except Exception as e:  # noqa: BLE001
                render_errors += 1
                log.debug("synth.render_err font=%s text=%r err=%s", font_path.name, text[:30], e)
                continue
            name = f"{idx:08d}.png"
            img.save(spec.out_dir / name)
            f.write(f"{name}\t{text}\n")
            written += 1
            if (idx + 1) % progress_every == 0:
                pct = (idx + 1) / spec.count * 100
                log.info("synth.progress %.0f%% written=%d font_err=%d render_err=%d",
                         pct, written, font_errors, render_errors,
                         extra={"op": "synth_recognition", "pct": pct,
                                "written": written})
    elapsed = _time.perf_counter() - started
    log.info(
        "synth.done written=%d font_err=%d render_err=%d elapsed_s=%.1f rate=%.0f/s out=%s",
        written, font_errors, render_errors, elapsed,
        written / max(elapsed, 1e-6), spec.out_dir,
        extra={"op": "synth_recognition", "event": "done",
               "written": written, "elapsed_s": elapsed},
    )
    return written


def default_korean_corpus() -> list[str]:
    """Small built-in corpus for smoke testing. Replace with kowiki for real runs."""
    samples = [
        "안녕하세요", "오늘 날씨가 맑음", "서울특별시 종로구",
        "상품 가격 12,345원", "배송 완료되었습니다", "회원가입 감사합니다",
        "영업시간 09:00~18:00", "대한민국 만세", "주소: 서울시 강남구",
        "SPINAI OCR v0.1", "Hello 세계 World", "전화번호 010-1234-5678",
        "이메일 test@example.com", "웹사이트 spinai.io", "한국어 최고!",
        "커피 4,500원 케이크 6,000원", "우편번호 06134", "국립중앙도서관",
        "주민등록증", "여권번호", "BIRTHDAY 2026-04-19",
    ]
    return samples
