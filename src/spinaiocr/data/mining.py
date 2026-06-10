"""Teacher-student divergence mining.

The goal the user specified: take (ideally) every web-scraped image, run
teacher OCRs (PaddleOCR + Gemma 4 + ...) and our own model on it, and surface
the samples where we disagree with the teacher consensus the most. Those
become the next training batch. Repeat.

This module implements the core loop:

    1. Load an image directory.
    2. For each image, compute:
         - teacher consensus (via :func:`spinaiocr.data.pseudo.consensus`)
         - our model's prediction (OCRPipeline)
         - divergence score (CER on joined text, polygon-IoU on boxes)
    3. Sort by divergence (high first) and write a priority JSONL.
    4. Optionally move the top-K into a `train/` partition and the rest
       into a `review/` partition for human QA.

Design notes:
    * We score divergence per-image (not per-line) because alignment across
      teachers is lossy. CER on the space-joined transcript is a good proxy.
    * Polygon disagreement is IoU-based: count how many teacher boxes we
      *miss* (IoU ≥ 0.5 with no matching student box).
    * Samples with zero teacher lines are skipped — they contribute no
      signal.
"""
from __future__ import annotations

import heapq
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from spinaiocr.benchmark.metrics import _levenshtein
from spinaiocr.data.pseudo import ConsensusConfig, consensus, polygon_iou
from spinaiocr.teachers.base import OCRTeacher


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _cer(hyp: str, ref: str) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(hyp, ref) / max(len(ref), 1)


def _box_miss_rate(
    teacher_polys: list[np.ndarray],
    student_polys: list[np.ndarray],
    iou_thresh: float = 0.5,
) -> float:
    if not teacher_polys:
        return 0.0
    missed = 0
    for tp in teacher_polys:
        best = max((polygon_iou(tp, sp) for sp in student_polys), default=0.0)
        if best < iou_thresh:
            missed += 1
    return missed / len(teacher_polys)


@dataclass
class DivergenceRecord:
    image: str
    text_cer: float
    box_miss: float
    n_teacher_words: int
    n_student_words: int
    teacher_text: str
    student_text: str
    teachers: list[str]
    priority: float  # higher = more valuable to retrain on


# ---------------------------------------------------------------------------
# Mining loop
# ---------------------------------------------------------------------------


@dataclass
class MiningConfig:
    iou_thresh: float = 0.5
    text_weight: float = 0.7
    box_weight: float = 0.3
    # Skip uninteresting extremes
    min_teacher_words: int = 1
    max_teacher_words: int = 200


def score_divergence(
    teacher_consensus_words: list[dict],
    student_lines,
    cfg: MiningConfig | None = None,
) -> DivergenceRecord | None:
    cfg = cfg or MiningConfig()
    if len(teacher_consensus_words) < cfg.min_teacher_words:
        return None
    if len(teacher_consensus_words) > cfg.max_teacher_words:
        return None

    teacher_text = " ".join(w["text"] for w in teacher_consensus_words)
    student_text = " ".join(l.text for l in student_lines)
    text_cer = _cer(student_text, teacher_text)

    teacher_polys = [np.array(w["points"], dtype=np.float32) for w in teacher_consensus_words]
    student_polys = [np.array(l.bbox, dtype=np.float32) for l in student_lines]
    box_miss = _box_miss_rate(teacher_polys, student_polys, cfg.iou_thresh)

    priority = cfg.text_weight * text_cer + cfg.box_weight * box_miss

    return DivergenceRecord(
        image="",
        text_cer=text_cer,
        box_miss=box_miss,
        n_teacher_words=len(teacher_consensus_words),
        n_student_words=len(student_lines),
        teacher_text=teacher_text,
        student_text=student_text,
        teachers=[],
        priority=priority,
    )


def mine_directory(
    image_dir: Path,
    teachers: list[OCRTeacher],
    student_pipeline,
    out_jsonl: Path,
    consensus_cfg: ConsensusConfig | None = None,
    cfg: MiningConfig | None = None,
    top_k: int | None = None,
    exts: Iterable[str] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
) -> dict:
    cfg = cfg or MiningConfig()
    consensus_cfg = consensus_cfg or ConsensusConfig()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    records: list[DivergenceRecord] = []
    top: list[tuple[float, int]] = []  # min-heap of (-priority, idx)

    for img in sorted(image_dir.rglob("*")):
        if img.suffix.lower() not in exts:
            continue
        try:
            preds = [t(img) for t in teachers]
            teacher_words = [
                {"points": w.points, "text": w.text}
                for w in consensus(preds, consensus_cfg)
            ]
            student = student_pipeline(str(img))
        except Exception:  # noqa: BLE001
            continue

        rec = score_divergence(teacher_words, student.lines, cfg)
        if rec is None:
            continue
        rec.image = str(img)
        rec.teachers = [p.teacher for p in preds]
        records.append(rec)

        if top_k is not None:
            heapq.heappush(top, (rec.priority, len(records) - 1))
            if len(top) > top_k:
                heapq.heappop(top)

    # write full log
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in sorted(records, key=lambda x: x.priority, reverse=True):
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    return {
        "total": len(records),
        "mean_priority": float(np.mean([r.priority for r in records])) if records else 0.0,
        "p90_priority": float(np.percentile([r.priority for r in records], 90)) if records else 0.0,
        "output": str(out_jsonl),
    }


def partition_by_priority(
    mining_jsonl: Path,
    priority_threshold: float,
    train_out: Path,
    review_out: Path,
) -> tuple[int, int]:
    """Split a mining log into a high-priority train set and a review queue."""
    train_count = 0
    review_count = 0
    with mining_jsonl.open("r", encoding="utf-8") as fin, \
         train_out.open("w", encoding="utf-8") as ftrain, \
         review_out.open("w", encoding="utf-8") as freview:
        for line in fin:
            rec = json.loads(line)
            target = ftrain if rec["priority"] >= priority_threshold else freview
            target.write(line)
            if target is ftrain:
                train_count += 1
            else:
                review_count += 1
    return train_count, review_count
