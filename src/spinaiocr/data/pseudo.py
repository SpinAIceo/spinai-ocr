"""Multi-teacher pseudo labeling with consensus voting.

Port of the medical imaging pipeline's `multi_teacher_pseudo.py` +
`two_model_agreement.py` + `sam3_ct_agreement.py`, generalized for OCR:

    Agreement rules
    ---------------
    Detection (bbox):
        A box is accepted iff at least ceil(N/2) teachers produced a box
        with IoU >= iou_thresh.

    Recognition (text):
        Among the hypotheses attached to matched boxes, pair-wise CER
        must be <= cer_thresh for a majority. The winning text is the
        mode of the agreeing teachers' strings.

    Confidence tagging:
        - `high`: all teachers agreed
        - `mid`:  >= ceil(N/2) agreed
        - `low`:  used only when `keep_low=True` (for manual review)

Outputs a JSONL manifest compatible with :class:`DetectionDataset`:

    {"image": "x.jpg", "words": [{"points": [...], "text": "...", "conf": 0.9}]}
"""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from shapely.geometry import Polygon

from spinaiocr.benchmark.metrics import _levenshtein
from spinaiocr.log import capture_crashes, get_logger
from spinaiocr.teachers.base import OCRTeacher, TeacherLine, TeacherPrediction

log = get_logger("spinaiocr.data.pseudo")


# ---------------------------------------------------------------------------
# Geometric helpers
# ---------------------------------------------------------------------------


def polygon_iou(a: np.ndarray, b: np.ndarray) -> float:
    pa = Polygon(a).buffer(0)
    pb = Polygon(b).buffer(0)
    if not pa.is_valid or not pb.is_valid or pa.is_empty or pb.is_empty:
        return 0.0
    inter = pa.intersection(pb).area
    union = pa.union(pb).area
    return float(inter / union) if union > 0 else 0.0


def _cer(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    return _levenshtein(a, b) / max(len(b), 1)


# ---------------------------------------------------------------------------
# Consensus
# ---------------------------------------------------------------------------


@dataclass
class ConsensusWord:
    points: list[list[float]]
    text: str
    confidence: float  # agreement ratio, 0~1
    tier: str  # "high" | "mid" | "low"
    voters: list[str]  # teacher names that contributed


@dataclass
class ConsensusConfig:
    iou_thresh: float = 0.5
    cer_thresh: float = 0.1
    min_voters: int = 2
    keep_low: bool = False


def _group_by_iou(preds: list[TeacherPrediction], iou_thresh: float) -> list[list[tuple[str, TeacherLine]]]:
    """Greedy bbox clustering across teachers (single-link)."""
    items = [(p.teacher, l) for p in preds for l in p.lines]
    groups: list[list[tuple[str, TeacherLine]]] = []
    for name, line in items:
        placed = False
        for g in groups:
            if any(polygon_iou(line.bbox, other.bbox) >= iou_thresh for _, other in g):
                g.append((name, line))
                placed = True
                break
        if not placed:
            groups.append([(name, line)])
    return groups


def _majority_text(entries: list[tuple[str, TeacherLine]], cer_thresh: float) -> tuple[str, list[str]]:
    """Return the majority text and list of voter teacher names."""
    # Start with the most common raw text; check CER agreement against it.
    texts = [e[1].text for e in entries]
    counter = Counter(texts)
    candidate, _ = counter.most_common(1)[0]
    voters = [e[0] for e in entries if _cer(e[1].text, candidate) <= cer_thresh]
    return candidate, voters


def consensus(
    preds: list[TeacherPrediction],
    cfg: ConsensusConfig | None = None,
) -> list[ConsensusWord]:
    cfg = cfg or ConsensusConfig()
    n_teachers = len(preds)
    if n_teachers == 0:
        return []
    min_agree = max(cfg.min_voters, math.ceil(n_teachers / 2))

    groups = _group_by_iou(preds, cfg.iou_thresh)
    out: list[ConsensusWord] = []
    for g in groups:
        if not g:
            continue
        text, voters = _majority_text(g, cfg.cer_thresh)
        n_agree = len(voters)
        if n_agree < min_agree and not cfg.keep_low:
            continue
        tier = (
            "high" if n_agree == n_teachers else ("mid" if n_agree >= min_agree else "low")
        )
        # average box from voters
        matched = [e for e in g if e[0] in voters] or g
        avg_bbox = np.mean(np.stack([e[1].bbox for e in matched]), axis=0)
        out.append(
            ConsensusWord(
                points=avg_bbox.tolist(),
                text=text,
                confidence=n_agree / n_teachers,
                tier=tier,
                voters=sorted(voters),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def pseudo_label_image(
    image_path: Path,
    teachers: list[OCRTeacher],
    cfg: ConsensusConfig | None = None,
) -> dict:
    preds = [t(image_path) for t in teachers]
    words = consensus(preds, cfg)
    return {
        "image": str(image_path),
        "words": [
            {
                "points": w.points,
                "text": w.text,
                "conf": w.confidence,
                "tier": w.tier,
                "voters": w.voters,
            }
            for w in words
        ],
        "teachers": [p.teacher for p in preds],
    }


def pseudo_label_directory(
    image_dir: Path,
    out_jsonl: Path,
    teachers: list[OCRTeacher],
    cfg: ConsensusConfig | None = None,
    exts: Iterable[str] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
) -> int:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    paths = [p for p in sorted(image_dir.rglob("*")) if p.suffix.lower() in exts]
    log.info("pseudo.start n_images=%d teachers=%s out=%s",
             len(paths), [t.name for t in teachers], str(out_jsonl),
             extra={"n_images": len(paths), "teachers": [t.name for t in teachers],
                    "out": str(out_jsonl)})
    count = 0
    errors = 0
    with out_jsonl.open("w", encoding="utf-8") as f:
        for img in paths:
            try:
                with capture_crashes("pseudo_label_image", extra={"image": str(img)},
                                     reraise=False):
                    record = pseudo_label_image(img, teachers, cfg)
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
            except Exception:
                errors += 1
    log.info("pseudo.done processed=%d errors=%d",
             count, errors, extra={"processed": count, "errors": errors})
    return count
