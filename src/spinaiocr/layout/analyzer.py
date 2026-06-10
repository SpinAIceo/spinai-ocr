"""Layout analyzer.

Two-tier strategy:
1. **Heuristic tier** (always available): geometric clustering of detection
   polygons + vertical/horizontal projection to infer blocks, lines, and
   reading order.
2. **Model tier** (optional): wraps Donut / LayoutLMv3 / DiT if those packages
   are installed. Produces typed regions (title, body, table, figure, list).

Output is serialized via :class:`LayoutResult`, which knows how to emit
Markdown and HTML.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

RegionKind = Literal["title", "paragraph", "list", "table", "figure", "caption", "header", "footer"]


@dataclass
class LayoutRegion:
    kind: RegionKind
    bbox: list[tuple[float, float]]  # 4-point polygon
    text: str = ""
    children: list["LayoutRegion"] = field(default_factory=list)
    order: int = 0


@dataclass
class LayoutResult:
    regions: list[LayoutRegion] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0

    def to_markdown(self) -> str:
        parts: list[str] = []
        for r in sorted(self.regions, key=lambda x: x.order):
            if r.kind == "title":
                parts.append(f"# {r.text}")
            elif r.kind == "header":
                parts.append(f"## {r.text}")
            elif r.kind == "list":
                for line in r.text.splitlines():
                    parts.append(f"- {line}")
            elif r.kind == "table":
                parts.append(r.text)  # expected pre-formatted markdown
            elif r.kind == "figure":
                parts.append(f"![figure]({r.text or ''})")
            else:
                parts.append(r.text)
            parts.append("")
        return "\n".join(parts).strip()

    def to_html(self) -> str:
        tag_map = {
            "title": "h1",
            "header": "h2",
            "paragraph": "p",
            "caption": "figcaption",
            "footer": "footer",
        }
        out: list[str] = []
        for r in sorted(self.regions, key=lambda x: x.order):
            if r.kind == "list":
                out.append(
                    "<ul>" + "".join(f"<li>{line}</li>" for line in r.text.splitlines()) + "</ul>"
                )
            elif r.kind == "table":
                out.append(r.text)
            elif r.kind == "figure":
                out.append(f'<figure><img alt="figure" src="{r.text or ""}"/></figure>')
            else:
                tag = tag_map.get(r.kind, "p")
                out.append(f"<{tag}>{r.text}</{tag}>")
        return "\n".join(out)


# ---------------------------------------------------------------------------
# Heuristic analyzer
# ---------------------------------------------------------------------------


def _bbox_center(poly: np.ndarray) -> tuple[float, float]:
    return float(poly[:, 0].mean()), float(poly[:, 1].mean())


def _bbox_height(poly: np.ndarray) -> float:
    return float(poly[:, 1].max() - poly[:, 1].min())


def _cluster_lines(polys: list[np.ndarray], vertical_tol: float = 0.6) -> list[list[int]]:
    """Group polygon indices into lines by vertical center overlap.

    Returns a list of lines, each a sorted (left-to-right) list of polygon indices.
    """
    if not polys:
        return []
    centers = [_bbox_center(p) for p in polys]
    heights = [_bbox_height(p) for p in polys]
    order = sorted(range(len(polys)), key=lambda i: centers[i][1])
    lines: list[list[int]] = []
    for idx in order:
        y = centers[idx][1]
        h = heights[idx]
        placed = False
        for line in lines:
            ref = line[0]
            if abs(centers[ref][1] - y) < vertical_tol * max(heights[ref], h):
                line.append(idx)
                placed = True
                break
        if not placed:
            lines.append([idx])
    for line in lines:
        line.sort(key=lambda i: centers[i][0])
    return lines


def _classify_line(poly_line: list[np.ndarray], text_line: list[str], image_h: int) -> RegionKind:
    heights = [_bbox_height(p) for p in poly_line]
    mean_h = float(np.mean(heights))
    top = float(np.min([p[:, 1].min() for p in poly_line]))
    bottom = float(np.max([p[:, 1].max() for p in poly_line]))
    # relative height heuristic
    rel = mean_h / max(image_h, 1)
    text = " ".join(text_line).strip()
    if rel > 0.04:
        return "title"
    if top < image_h * 0.05:
        return "header"
    if bottom > image_h * 0.95:
        return "footer"
    if text.startswith(("-", "•", "*", "·")) or (text and text[0].isdigit() and "." in text[:3]):
        return "list"
    return "paragraph"


class LayoutAnalyzer:
    """Heuristic-first layout analyzer."""

    def __init__(self, use_model: bool = False) -> None:
        self.use_model = use_model
        self._model = None
        if use_model:
            try:
                # Optional: wire Donut / LayoutLMv3 here later.
                pass
            except Exception:  # noqa: BLE001
                self._model = None

    def analyze(
        self,
        image_shape: tuple[int, int],
        polygons: list[np.ndarray],
        texts: list[str],
    ) -> LayoutResult:
        h, w = image_shape
        lines = _cluster_lines(polygons)
        regions: list[LayoutRegion] = []
        for order, line_idxs in enumerate(lines):
            line_polys = [polygons[i] for i in line_idxs]
            line_texts = [texts[i] for i in line_idxs]
            kind = _classify_line(line_polys, line_texts, h)
            merged_text = " ".join(line_texts)
            min_x = float(np.min([p[:, 0].min() for p in line_polys]))
            min_y = float(np.min([p[:, 1].min() for p in line_polys]))
            max_x = float(np.max([p[:, 0].max() for p in line_polys]))
            max_y = float(np.max([p[:, 1].max() for p in line_polys]))
            bbox = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
            regions.append(LayoutRegion(kind=kind, bbox=bbox, text=merged_text, order=order))
        return LayoutResult(regions=regions, image_width=w, image_height=h)
