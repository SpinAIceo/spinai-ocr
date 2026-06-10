"""Polygon-aware augmentations for text detection training.

These are separate from the recognition-only crop augmentations in
`augment.py` because the polygon annotations must follow the image
through every geometric transform.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np


Polygons = list[np.ndarray]


def _transform_polys(polys: Polygons, M: np.ndarray) -> Polygons:
    out = []
    for p in polys:
        pts = np.concatenate([p, np.ones((p.shape[0], 1), dtype=np.float32)], axis=1)
        warped = (pts @ M.T)[:, :2]
        out.append(warped.astype(np.float32))
    return out


def random_rotate_with_polys(
    image: np.ndarray, polys: Polygons, max_deg: float = 10.0, border_value: int = 255
) -> tuple[np.ndarray, Polygons]:
    h, w = image.shape[:2]
    angle = random.uniform(-max_deg, max_deg)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(image, M, (w, h), borderValue=(border_value,) * 3)
    M3 = np.eye(3, dtype=np.float32)
    M3[:2] = M
    return img, _transform_polys(polys, M3)


def random_scale_with_polys(
    image: np.ndarray, polys: Polygons, scale_range: tuple[float, float] = (0.7, 1.3)
) -> tuple[np.ndarray, Polygons]:
    s = random.uniform(*scale_range)
    h, w = image.shape[:2]
    nh, nw = int(h * s), int(w * s)
    img = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    M = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], dtype=np.float32)
    return img, _transform_polys(polys, M)


def random_crop_with_polys(
    image: np.ndarray,
    polys: Polygons,
    texts: list[str],
    size: int = 640,
    min_polygon_in_crop: int = 1,
    max_tries: int = 20,
) -> tuple[np.ndarray, Polygons, list[str]]:
    h, w = image.shape[:2]
    target = min(size, h, w)
    if h == target and w == target:
        return image, polys, texts

    for _ in range(max_tries):
        x = random.randint(0, max(w - target, 0))
        y = random.randint(0, max(h - target, 0))
        crop = image[y : y + target, x : x + target]
        kept_polys: Polygons = []
        kept_texts: list[str] = []
        for p, t in zip(polys, texts):
            shifted = p - np.array([x, y], dtype=np.float32)
            # Keep only polygons fully inside the crop
            if (
                (shifted[:, 0] >= 0).all()
                and (shifted[:, 0] < target).all()
                and (shifted[:, 1] >= 0).all()
                and (shifted[:, 1] < target).all()
            ):
                kept_polys.append(shifted)
                kept_texts.append(t)
        if len(kept_polys) >= min_polygon_in_crop:
            return crop, kept_polys, kept_texts
    return image, polys, texts


def horizontal_flip_with_polys(
    image: np.ndarray, polys: Polygons
) -> tuple[np.ndarray, Polygons]:
    # Rarely used for OCR (text mirror-reverses unnaturally), provided for completeness.
    h, w = image.shape[:2]
    img = cv2.flip(image, 1)
    flipped = []
    for p in polys:
        q = p.copy()
        q[:, 0] = w - 1 - q[:, 0]
        flipped.append(q)
    return img, flipped


@dataclass
class _Op:
    fn: Callable
    p: float


class DetectionAugment:
    """Probability-gated chain for detection training."""

    def __init__(self, ops: list[_Op]) -> None:
        self.ops = ops

    def __call__(
        self, image: np.ndarray, polys: Polygons, texts: list[str]
    ) -> tuple[np.ndarray, Polygons, list[str]]:
        for op in self.ops:
            if random.random() >= op.p:
                continue
            result = op.fn(image, polys, texts) if op.fn.__code__.co_argcount == 3 else op.fn(image, polys)
            if len(result) == 3:
                image, polys, texts = result
            else:
                image, polys = result
        return image, polys, texts


def default_detection_augment() -> DetectionAugment:
    return DetectionAugment(
        [
            _Op(random_rotate_with_polys, 0.5),
            _Op(random_scale_with_polys, 0.5),
            _Op(random_crop_with_polys, 0.8),
        ]
    )
