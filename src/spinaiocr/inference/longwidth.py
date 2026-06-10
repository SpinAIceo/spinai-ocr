"""Sliding-window recognition for crops wider than the model's max_width.

The SVTR recognizer is trained at max_width=320. Wider inputs get truncated,
which loses text. This module splits a wide image into overlapping windows,
runs recognition on each, and merges the predictions.

Strategy (text-level merge):
    1. Tile the image with stride = window // 2 (50% overlap).
    2. Run recognition → text per tile.
    3. Merge by finding the longest suffix of tile[i-1] that is a prefix of
       tile[i], then append the rest of tile[i]. Fallback to concat when no
       overlap match exists (tile boundary inside a character).

This is lossy but fixes the common "25000px subtitle" case where the previous
behavior returned only the first ~10 characters.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def _overlap_merge(a: str, b: str, min_overlap: int = 1, max_overlap: int | None = None) -> str:
    """Merge b onto a by finding the longest suffix of a that is a prefix of b.

    If no overlap of ≥ `min_overlap` chars found, concatenate directly.
    """
    if not a:
        return b
    if not b:
        return a
    if max_overlap is None:
        max_overlap = min(len(a), len(b))
    # Try from longest to shortest; return on first match
    for k in range(max_overlap, min_overlap - 1, -1):
        if a.endswith(b[:k]):
            return a + b[k:]
    return a + b


def tile_image(img: np.ndarray, window: int, stride: int | None = None) -> list[tuple[int, np.ndarray]]:
    """Split a [H, W, C] array into ≤window-wide tiles. Returns (x_offset, tile).

    Final tile is right-anchored to ensure coverage.
    """
    h, w = img.shape[:2]
    if w <= window:
        return [(0, img)]
    stride = stride or window // 2
    tiles: list[tuple[int, np.ndarray]] = []
    x = 0
    while x + window < w:
        tiles.append((x, img[:, x : x + window]))
        x += stride
    # final right-anchored tile so the last characters aren't lost
    tiles.append((w - window, img[:, w - window : w]))
    return tiles


def recognize_long(
    img: np.ndarray,
    recognize_fn: Callable[[np.ndarray], str],
    window: int = 320,
    stride: int | None = None,
) -> str:
    """Run `recognize_fn` over sliding windows and merge outputs by overlap."""
    tiles = tile_image(img, window=window, stride=stride)
    if len(tiles) == 1:
        return recognize_fn(tiles[0][1])
    parts = [recognize_fn(t) for _, t in tiles]
    merged = parts[0]
    for p in parts[1:]:
        merged = _overlap_merge(merged, p)
    return merged
