"""Curriculum sampler.

Present easier examples first, gradually expose the model to harder ones.
Difficulty score is a weighted sum of:
    * text length (longer = harder)
    * rare character fraction (normalized by per-char vocab frequency)
    * image contrast (low contrast = harder)

At step `t` out of `total_steps` we sample from the bottom `frac(t)` of the
difficulty-sorted set, where `frac(t)` linearly grows from `start_frac` to
1.0 by the `warmup_frac * total_steps` mark.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from torch.utils.data import Sampler


@dataclass
class CurriculumConfig:
    total_steps: int
    start_frac: float = 0.2      # initially sample from easiest 20%
    warmup_frac: float = 0.3     # reach full set at 30% of total_steps
    len_weight: float = 1.0
    rare_weight: float = 1.5
    contrast_weight: float = 0.5


def _difficulty_scores(
    texts: list[str],
    image_paths: list | None = None,
    *,
    cfg: CurriculumConfig,
) -> np.ndarray:
    lens = np.array([len(t) for t in texts], dtype=np.float32)
    lens = (lens - lens.min()) / max(lens.max() - lens.min(), 1e-6)

    # rare-char fraction
    char_freq: Counter = Counter()
    for t in texts:
        char_freq.update(t)
    total = sum(char_freq.values())
    rarity = np.array([
        sum(1 for c in t if char_freq[c] / max(total, 1) < 0.01) / max(len(t), 1)
        for t in texts
    ], dtype=np.float32)
    # rarity already in [0,1]

    contrast = np.zeros_like(lens)
    if image_paths is not None:
        try:
            import cv2
        except ImportError:
            cv2 = None
        if cv2 is not None:
            for i, p in enumerate(image_paths):
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    contrast[i] = 0.5
                    continue
                c = img.std() / 255.0  # roughly [0, 0.5]
                contrast[i] = 1.0 - min(1.0, c * 2)  # low contrast → high difficulty
    return (
        cfg.len_weight * lens
        + cfg.rare_weight * rarity
        + cfg.contrast_weight * contrast
    )


class CurriculumSampler(Sampler):
    """Index sampler yielding indices sorted by difficulty and windowed by step."""

    def __init__(
        self,
        difficulty: np.ndarray,
        cfg: CurriculumConfig,
        seed: int = 0,
    ) -> None:
        self.cfg = cfg
        self._order = np.argsort(difficulty)  # easiest first
        self._n = len(self._order)
        self._rng = np.random.default_rng(seed)
        self._step = 0

    def set_step(self, step: int) -> None:
        self._step = step

    def __iter__(self):
        while True:
            progress = min(self._step / max(self.cfg.warmup_frac * self.cfg.total_steps, 1), 1.0)
            frac = self.cfg.start_frac + (1.0 - self.cfg.start_frac) * progress
            k = max(1, int(self._n * frac))
            window = self._order[:k]
            idx = int(self._rng.choice(window))
            yield idx
            self._step += 1

    def __len__(self) -> int:
        return self.cfg.total_steps


def build_curriculum_sampler(
    texts: Iterable[str],
    image_paths: Iterable | None,
    cfg: CurriculumConfig,
    seed: int = 0,
) -> CurriculumSampler:
    texts = list(texts)
    image_paths_list = list(image_paths) if image_paths is not None else None
    difficulty = _difficulty_scores(texts, image_paths_list, cfg=cfg)
    return CurriculumSampler(difficulty, cfg, seed=seed)
