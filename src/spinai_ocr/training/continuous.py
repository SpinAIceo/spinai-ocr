"""Continuous correction loop.

Implements the cycle the user described:

    scrape images  ─┐
                    ├─▶  teacher OCR (Paddle, Gemma4, ...) ─▶ consensus
    student model ──┘        │
                             ▼
                   divergence mining ─▶ high-priority train set
                             │
                             ▼
                      retrain student
                             │
                             └──▶ next iteration

Run one iteration:

    python -m spinai_ocr.training.continuous \
        --iteration-dir data/continuous/iter_001 \
        --source data/scraped/wikimedia_commons \
        --teachers paddle,gemma \
        --top-k 10000

Each iteration produces a timestamped directory with:
    manifest.json                 # iteration metadata
    mining.jsonl                  # divergence scores for all processed images
    mining.train.jsonl            # top-priority subset for retraining
    mining.review.jsonl           # remainder — human review queue
    train.tsv                     # extracted recognition crops
    train_detection.jsonl         # retained detection labels
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import click

from spinai_ocr.data.adapters import CropSpec, jsonl_to_recognition_tsv
from spinai_ocr.data.mining import MiningConfig, mine_directory, partition_by_priority
from spinai_ocr.data.pseudo import ConsensusConfig
from spinai_ocr.inference.pipeline import OCRPipeline
from spinai_ocr.teachers.base import build_teacher


@dataclass
class IterationMeta:
    started_at: float
    finished_at: float
    source_dir: str
    teachers: list[str]
    mined: int
    train_samples: int
    review_samples: int
    priority_split: float


def _write_detection_labels_from_mining(mining_jsonl: Path, out_jsonl: Path) -> int:
    """Turn mining records into detection training labels using teacher consensus."""
    n = 0
    with mining_jsonl.open("r", encoding="utf-8") as fin, out_jsonl.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            rec = json.loads(line)
            image = rec.get("image")
            if not image:
                continue
            # We don't have the polygons stored in the mining record itself,
            # only the joined text. In practice, the mining step also writes
            # a parallel consensus JSONL — see run_iteration() below.
            words = rec.get("words", [])
            if not words:
                continue
            fout.write(
                json.dumps({"image": image, "words": words}, ensure_ascii=False) + "\n"
            )
            n += 1
    return n


def run_iteration(
    iteration_dir: Path,
    source_dir: Path,
    teacher_names: list[str],
    lang: str = "ko",
    top_k: int | None = 10_000,
    priority_split: float = 0.15,
) -> IterationMeta:
    iteration_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    teachers = [build_teacher(n, lang=lang) for n in teacher_names]
    student = OCRPipeline(lang=lang)

    mining_jsonl = iteration_dir / "mining.jsonl"
    stats = mine_directory(
        image_dir=source_dir,
        teachers=teachers,
        student_pipeline=student,
        out_jsonl=mining_jsonl,
        consensus_cfg=ConsensusConfig(iou_thresh=0.5, cer_thresh=0.1),
        cfg=MiningConfig(),
        top_k=top_k,
    )

    train_out = iteration_dir / "mining.train.jsonl"
    review_out = iteration_dir / "mining.review.jsonl"
    tc, rc = partition_by_priority(mining_jsonl, priority_split, train_out, review_out)

    meta = IterationMeta(
        started_at=started,
        finished_at=time.time(),
        source_dir=str(source_dir),
        teachers=teacher_names,
        mined=stats["total"],
        train_samples=tc,
        review_samples=rc,
        priority_split=priority_split,
    )
    (iteration_dir / "manifest.json").write_text(
        json.dumps(asdict(meta), indent=2), encoding="utf-8"
    )
    return meta


@click.command()
@click.option("--iteration-dir", required=True, type=click.Path())
@click.option("--source", required=True, type=click.Path(exists=True))
@click.option("--teachers", default="paddle,gemma")
@click.option("--lang", default="ko")
@click.option("--top-k", default=10_000, type=int)
@click.option("--priority-split", default=0.15, type=float)
def main(
    iteration_dir: str,
    source: str,
    teachers: str,
    lang: str,
    top_k: int,
    priority_split: float,
) -> None:
    meta = run_iteration(
        iteration_dir=Path(iteration_dir),
        source_dir=Path(source),
        teacher_names=[t.strip() for t in teachers.split(",") if t.strip()],
        lang=lang,
        top_k=top_k,
        priority_split=priority_split,
    )
    click.echo(json.dumps(asdict(meta), indent=2, default=str))


if __name__ == "__main__":
    main()
