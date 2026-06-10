"""DBNet ground-truth generator.

Given polygon annotations, produce the four tensors DBNet needs:
- prob_map:  binary shrunk-polygon mask (probability target)
- prob_mask: valid-pixel mask (1 everywhere except `ignore` polygons)
- thresh_map: distance-weighted boundary band (per-pixel threshold target)
- thresh_mask: where `thresh_map` is valid (1 inside the band, 0 elsewhere)

References:
- Differentiable Binarization (Liao et al., 2019).
- PaddleOCR's `make_border_map.py` / `make_shrink_map.py`.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon


@dataclass
class DBGTConfig:
    shrink_ratio: float = 0.4  # DB uses 0.4 (r in the paper)
    thresh_min: float = 0.3
    thresh_max: float = 0.7


def _polygon_shrink_distance(poly: np.ndarray, shrink_ratio: float) -> float:
    """D = A * (1 - r^2) / L, per Vatti clipping recipe used by DBNet."""
    sh = Polygon(poly)
    if sh.length == 0:
        return 0.0
    return sh.area * (1 - shrink_ratio**2) / sh.length


def _offset_polygon(poly: np.ndarray, distance: float) -> list[np.ndarray]:
    pco = pyclipper.PyclipperOffset()
    pco.AddPath(poly.astype(np.int32).tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    result = pco.Execute(distance)
    return [np.array(r, dtype=np.int32) for r in result]


def make_shrink_map(
    size: tuple[int, int],
    polygons: list[np.ndarray],
    ignore_flags: list[bool] | None = None,
    shrink_ratio: float = 0.4,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = size
    prob_map = np.zeros((h, w), dtype=np.float32)
    prob_mask = np.ones((h, w), dtype=np.float32)
    if ignore_flags is None:
        ignore_flags = [False] * len(polygons)

    for poly, ignore in zip(polygons, ignore_flags):
        poly = np.asarray(poly)
        if poly.shape[0] < 3 or Polygon(poly).area < 1:
            cv2.fillPoly(prob_mask, [poly.astype(np.int32)], 0)
            continue
        if ignore:
            cv2.fillPoly(prob_mask, [poly.astype(np.int32)], 0)
            continue
        d = _polygon_shrink_distance(poly, shrink_ratio)
        shrunk = _offset_polygon(poly, -d)
        for s in shrunk:
            cv2.fillPoly(prob_map, [s], 1)
    return prob_map, prob_mask


def make_thresh_map(
    size: tuple[int, int],
    polygons: list[np.ndarray],
    ignore_flags: list[bool] | None = None,
    shrink_ratio: float = 0.4,
    thresh_min: float = 0.3,
    thresh_max: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    """Distance-transform-based threshold map around each polygon boundary."""
    h, w = size
    thresh_map = np.zeros((h, w), dtype=np.float32)
    thresh_mask = np.zeros((h, w), dtype=np.float32)
    if ignore_flags is None:
        ignore_flags = [False] * len(polygons)

    for poly, ignore in zip(polygons, ignore_flags):
        if ignore:
            continue
        poly = np.asarray(poly)
        if poly.shape[0] < 3 or Polygon(poly).area < 1:
            continue
        d = _polygon_shrink_distance(poly, shrink_ratio)
        expanded = _offset_polygon(poly, d)
        if not expanded:
            continue

        # region-of-interest bounding box
        expanded_poly = expanded[0]
        x, y, rw, rh = cv2.boundingRect(expanded_poly)
        x1, y1 = max(x, 0), max(y, 0)
        x2, y2 = min(x + rw, w), min(y + rh, h)
        if x2 <= x1 or y2 <= y1:
            continue

        # O(HW) distance via cv2.distanceTransform on the polygon edge mask.
        # Draw polygon outline (1-px thick) inside ROI, then distanceTransform
        # reports unsigned Euclidean distance to the nearest edge pixel.
        roi_h, roi_w = y2 - y1, x2 - x1
        roi_poly = (poly - np.array([x1, y1])).astype(np.int32)
        edge_mask = np.ones((roi_h, roi_w), dtype=np.uint8)
        cv2.polylines(edge_mask, [roi_poly], isClosed=True, color=0, thickness=1)
        dist = cv2.distanceTransform(edge_mask, cv2.DIST_L2, maskSize=3)
        dist = dist / max(d, 1e-6)
        dist = np.clip(1 - dist, 0, 1)

        sub = thresh_map[y1:y2, x1:x2]
        np.maximum(sub, dist, out=sub)
        cv2.fillPoly(thresh_mask[y1:y2, x1:x2], [expanded_poly - np.array([x1, y1])], 1)

    thresh_map = thresh_map * (thresh_max - thresh_min) + thresh_min
    return thresh_map, thresh_mask


def build_dbnet_targets(
    image_size: tuple[int, int],
    polygons: list[np.ndarray],
    ignore_flags: list[bool] | None = None,
    cfg: DBGTConfig | None = None,
) -> dict[str, np.ndarray]:
    cfg = cfg or DBGTConfig()
    prob_map, prob_mask = make_shrink_map(
        image_size, polygons, ignore_flags, cfg.shrink_ratio
    )
    thresh_map, thresh_mask = make_thresh_map(
        image_size, polygons, ignore_flags, cfg.shrink_ratio, cfg.thresh_min, cfg.thresh_max
    )
    return {
        "prob_map": prob_map,
        "prob_mask": prob_mask,
        "thresh_map": thresh_map,
        "thresh_mask": thresh_mask,
    }
