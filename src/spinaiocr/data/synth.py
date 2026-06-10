"""Synthetic training data generator wrapper.

Delegates heavy lifting to `trdg` and `synthtiger` (installed via `[synth]` extra).
We keep our own wrapper so training configs can reference SPINAI-local fonts and
corpora (Korean wiki dumps, news headlines, product names, etc.) deterministically.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DEFAULT_FONTS_DIR = Path(__file__).parent.parent / "assets" / "fonts"
DEFAULT_CORPUS_DIR = Path(__file__).parent.parent / "assets" / "corpora"


@dataclass
class SynthSpec:
    lang: str = "ko"
    count: int = 100_000
    out_dir: Path = Path("data/synthetic/ko")
    image_height: int = 48
    max_chars: int = 25
    fonts_dir: Path = DEFAULT_FONTS_DIR
    corpus_files: list[Path] | None = None
    distortions: bool = True
    backgrounds: bool = True


def _iter_corpus_lines(corpus_files: Iterable[Path], max_chars: int) -> Iterable[str]:
    for path in corpus_files:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield line[:max_chars]


def generate(spec: SynthSpec) -> int:
    """Generate synthetic recognition samples. Returns number of images written.

    Note: actual rendering requires `trdg` at runtime. If unavailable, this
    function writes a spec file only (for CI smoke tests).
    """
    spec.out_dir.mkdir(parents=True, exist_ok=True)
    spec_path = spec.out_dir / "spec.json"
    spec_path.write_text(
        "\n".join(
            [
                f"lang={spec.lang}",
                f"count={spec.count}",
                f"image_height={spec.image_height}",
                f"max_chars={spec.max_chars}",
                f"distortions={spec.distortions}",
                f"backgrounds={spec.backgrounds}",
            ]
        ),
        encoding="utf-8",
    )

    try:
        from trdg.generators import GeneratorFromStrings  # type: ignore
    except ImportError:
        # Soft fail — full generation needs `pip install -e '.[synth]'`
        return 0

    corpus_files = spec.corpus_files or []
    strings = list(_iter_corpus_lines(corpus_files, spec.max_chars))
    if not strings:
        # fallback: random short Korean syllables
        strings = ["".join(chr(random.randint(0xAC00, 0xD7A3)) for _ in range(random.randint(2, 8))) for _ in range(1000)]

    gen = GeneratorFromStrings(
        strings=strings[: spec.count],
        count=spec.count,
        size=spec.image_height,
        skewing_angle=3 if spec.distortions else 0,
        random_skew=spec.distortions,
        blur=1 if spec.distortions else 0,
        random_blur=spec.distortions,
        background_type=random.choice([0, 1, 2, 3]) if spec.backgrounds else 0,
        language="ko" if spec.lang == "ko" else "en",
    )

    label_file = spec.out_dir / "labels.txt"
    written = 0
    with label_file.open("w", encoding="utf-8") as lf:
        for idx, (img, text) in enumerate(gen):
            if img is None:
                continue
            filename = f"{idx:08d}.png"
            img.save(spec.out_dir / filename)
            lf.write(f"{filename}\t{text}\n")
            written += 1
            if written >= spec.count:
                break
    return written
