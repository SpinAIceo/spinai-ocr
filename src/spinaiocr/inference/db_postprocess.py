"""DBNet post-processor: prob map → polygon boxes.

Steps:
    1. Binarize prob map at `thresh`.
    2. Find contours.
    3. Filter by `box_thresh` using the mean prob inside each candidate.
    4. Expand (unclip) by a ratio to recover the original text region.
    5. Return 4-point polygons (min-area rectangle approximation).
"""
from __future__ import annotations

import cv2
import numpy as np
import pyclipper


class DBPostProcessor:
    def __init__(
        self,
        thresh: float = 0.3,
        box_thresh: float = 0.6,
        unclip_ratio: float = 1.5,
        max_candidates: int = 1000,
        min_box_size: int = 3,
        # iter 80: aspect-ratio-aware horizontal unclip extension. After uniform
        # pyclipper offset, optionally expand the box horizontally by an extra
        # multiplier when aspect ratio ≥ wide_ar_thresh. iter 79 audit found
        # 97% of detection misses are wide ar≥3 text where uniform unclip
        # under-extends horizontally relative to GT span.
        wide_ar_thresh: float = 3.0,
        wide_horizontal_extra: float = 0.0,
        # iter 113: secondary dilation for rec-crop only (decoupling). After
        # extract_polygons returns tight polys (unclip_ratio=2.0 for F1
        # optimum), pipeline calls dilate_for_recognize(poly) before warping
        # to rec input. Equivalent to `unclip(poly, recognize_extra_ratio)`
        # applied as a chained second pass.
        recognize_extra_ratio: float = 0.0,
    ) -> None:
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.unclip_ratio = unclip_ratio
        self.max_candidates = max_candidates
        self.min_box_size = min_box_size
        self.wide_ar_thresh = wide_ar_thresh
        self.wide_horizontal_extra = wide_horizontal_extra
        self.recognize_extra_ratio = recognize_extra_ratio

    def _expand_horizontal(self, box: np.ndarray, extra: float) -> np.ndarray:
        """Expand the 4-point box horizontally by `extra` fraction on each side.
        Box order: [tl, tr, br, bl]. tl-bl share x, tr-br share x (axis-aligned approx)."""
        if extra <= 0:
            return box
        cx = (box[:, 0].min() + box[:, 0].max()) / 2.0
        out = box.copy()
        for i, pt in enumerate(box):
            dx = pt[0] - cx
            out[i, 0] = cx + dx * (1.0 + extra)
        return out

    def _unclip(self, poly: np.ndarray) -> list[np.ndarray]:
        # iter 105: shoelace area + np.linalg.norm perimeter, byte-identical
        # to shapely Polygon(poly).area / .length on a 4-point box. Drops
        # the shapely dependency from this module.
        pts = np.asarray(poly, dtype=np.float64)
        x = pts[:, 0]
        y = pts[:, 1]
        area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)))
        diffs = np.diff(np.vstack([pts, pts[:1]]), axis=0)
        perim = float(np.linalg.norm(diffs, axis=1).sum())
        if perim == 0:
            return []
        distance = area * self.unclip_ratio / perim
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(poly.astype(np.int32).tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = pco.Execute(distance)
        return [np.array(e, dtype=np.int32) for e in expanded]

    def dilate_for_recognize(self, poly: np.ndarray) -> np.ndarray:
        """iter 113: chained dilation for rec-crop decoupling.

        Returns a 4-point quad expanded from `poly` by
        `recognize_extra_ratio * area / perim`. Used by pipeline before
        `_crop_polygon` to give the recognizer a comfortable crop while the
        returned/F1-scored polygon stays tight (unclip_ratio=2.0 result).

        If `recognize_extra_ratio == 0.0`: returns poly unchanged (back-compat).
        On degenerate input (zero perim/area or pyclipper failure): returns
        poly unchanged — caller observes no behavior change.
        """
        extra = self.recognize_extra_ratio
        if extra <= 0.0:
            return poly
        pts = np.asarray(poly, dtype=np.float64)
        x = pts[:, 0]; y = pts[:, 1]
        area = 0.5 * abs(float(
            np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)))
        diffs = np.diff(np.vstack([pts, pts[:1]]), axis=0)
        perim = float(np.linalg.norm(diffs, axis=1).sum())
        if perim == 0.0 or area == 0.0:
            return poly
        distance = area * extra / perim
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(poly.astype(np.int32).tolist(),
                    pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = pco.Execute(distance)
        if not expanded:
            return poly
        return self._min_area_rect(np.array(expanded[0], dtype=np.int32))

    def _min_area_rect(self, contour: np.ndarray) -> np.ndarray:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        # order: top-left, top-right, bottom-right, bottom-left
        order = np.argsort(box[:, 0])
        left = box[order[:2]]
        right = box[order[2:]]
        left = left[np.argsort(left[:, 1])]
        right = right[np.argsort(right[:, 1])]
        return np.array([left[0], right[0], right[1], left[1]], dtype=np.float32)

    def _box_score(self, prob_map: np.ndarray, contour: np.ndarray) -> float:
        h, w = prob_map.shape
        xmin = int(np.clip(contour[:, 0].min(), 0, w - 1))
        xmax = int(np.clip(contour[:, 0].max(), 0, w - 1))
        ymin = int(np.clip(contour[:, 1].min(), 0, h - 1))
        ymax = int(np.clip(contour[:, 1].max(), 0, h - 1))
        if xmax <= xmin or ymax <= ymin:
            return 0.0
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        shifted = contour.copy()
        shifted[:, 0] -= xmin
        shifted[:, 1] -= ymin
        cv2.fillPoly(mask, [shifted.astype(np.int32)], 1)
        return float(cv2.mean(prob_map[ymin : ymax + 1, xmin : xmax + 1], mask=mask)[0])

    def extract_polygons(self, prob_map: np.ndarray, image_size: tuple[int, int]) -> list[np.ndarray]:
        """prob_map: HxW float in [0,1]. Returns list of Nx2 polygons in pixel coords."""
        return [p for p, _ in self.extract_polygons_with_scores(prob_map, image_size)]

    def extract_polygons_with_scores(
        self, prob_map: np.ndarray, image_size: tuple[int, int]
    ) -> list[tuple[np.ndarray, float]]:
        """Same as extract_polygons but also returns the mean-prob score per box.
        Score can feed downstream filters (e.g. skip recognition on low-score crops)."""
        binary = (prob_map > self.thresh).astype(np.uint8) * 255
        # iter 98: RETR_EXTERNAL skips nested child contours. DBNet's
        # prob_map is text/no-text at the line/word level — interior holes
        # (Korean ㅁ/ㅇ/ㅂ counters; prob dips inside a high-prob region)
        # are not separate text. Quality A-B (n=50 synth det_ko, byte-
        # identical text + n_polys=354/354/0-diff) confirms no regression.
        # Tiny perf benefit (-1-2% of post_det = ~20µs) and simpler contour
        # tree. Was RETR_LIST.
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[: self.max_candidates]

        h, w = image_size
        out: list[tuple[np.ndarray, float]] = []
        for cnt in contours:
            if cnt.shape[0] < 4:
                continue
            c = cnt.reshape(-1, 2)
            score = self._box_score(prob_map, c)
            if score < self.box_thresh:
                continue
            box = self._min_area_rect(c)
            side = min(
                np.linalg.norm(box[0] - box[1]),
                np.linalg.norm(box[1] - box[2]),
            )
            if side < self.min_box_size:
                continue
            expanded = self._unclip(box)
            if not expanded:
                continue
            final = self._min_area_rect(expanded[0])
            # iter 80: width-aware unclip — additional horizontal expansion on
            # wide-aspect boxes (iter 79 audit: 97% of misses are wide ar≥3).
            if self.wide_horizontal_extra > 0:
                bw = max(final[:, 0].max() - final[:, 0].min(), 1.0)
                bh = max(final[:, 1].max() - final[:, 1].min(), 1.0)
                if bw / bh >= self.wide_ar_thresh:
                    final = self._expand_horizontal(final, self.wide_horizontal_extra)
            final[:, 0] = np.clip(final[:, 0], 0, w - 1)
            final[:, 1] = np.clip(final[:, 1], 0, h - 1)
            out.append((final, float(score)))
        return out
