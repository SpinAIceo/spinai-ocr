"""Table structure recognition.

Heuristic version now, model-based (PubTabNet-style TableMaster / TATR) later.

The heuristic works on an already-OCR'd image:
    Input: list of (polygon, text) tuples inside a detected table region.
    Output: a 2-D grid of cell texts, serialized to Markdown/HTML.

Algorithm:
    1. Project centers onto the y-axis → cluster into rows.
    2. Within each row, sort by x and bin into columns using k-means-like
       split on x-centers across rows.
    3. Emit Markdown table with `|` separators.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TableResult:
    cells: list[list[str]]  # rows × cols

    def to_markdown(self) -> str:
        if not self.cells:
            return ""
        header = self.cells[0]
        sep = ["---"] * len(header)
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(sep) + " |",
        ]
        for row in self.cells[1:]:
            # pad short rows
            padded = row + [""] * (len(header) - len(row))
            lines.append("| " + " | ".join(padded[: len(header)]) + " |")
        return "\n".join(lines)

    def to_html(self) -> str:
        rows = []
        for i, row in enumerate(self.cells):
            tag = "th" if i == 0 else "td"
            cells_html = "".join(f"<{tag}>{c}</{tag}>" for c in row)
            rows.append(f"<tr>{cells_html}</tr>")
        return "<table>" + "".join(rows) + "</table>"


def _cluster_rows(polys: list[np.ndarray], tol: float = 0.6) -> list[list[int]]:
    centers_y = [(p[:, 1].mean(), p[:, 1].max() - p[:, 1].min()) for p in polys]
    order = sorted(range(len(polys)), key=lambda i: centers_y[i][0])
    rows: list[list[int]] = []
    for idx in order:
        y, h = centers_y[idx]
        placed = False
        for row in rows:
            ry = centers_y[row[0]][0]
            rh = centers_y[row[0]][1]
            if abs(y - ry) < tol * max(h, rh):
                row.append(idx)
                placed = True
                break
        if not placed:
            rows.append([idx])
    for row in rows:
        row.sort(key=lambda i: polys[i][:, 0].mean())
    return rows


def _cluster_columns(rows: list[list[int]], polys: list[np.ndarray], n_cols: int | None = None) -> int:
    if n_cols is not None:
        return n_cols
    # Use the max cells-per-row as the column count estimate
    return max(len(r) for r in rows) if rows else 0


def recognize_table(
    polys: list[np.ndarray],
    texts: list[str],
    n_cols: int | None = None,
) -> TableResult:
    if not polys:
        return TableResult(cells=[])
    rows = _cluster_rows(polys)
    cols = _cluster_columns(rows, polys, n_cols)
    grid: list[list[str]] = []
    for row in rows:
        line = [texts[i] for i in row]
        # pad or truncate to column count
        line = line + [""] * (cols - len(line))
        grid.append(line[:cols])
    return TableResult(cells=grid)
