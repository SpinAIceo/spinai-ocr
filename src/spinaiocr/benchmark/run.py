"""Benchmark runner.

Compares SPINAI OCR against a labeled dataset and optionally against a competitor
(PaddleOCR, EasyOCR) by running each engine on the same images.

Outputs JSON + Markdown report to `benchmarks/reports/`.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import click

from spinaiocr.benchmark.metrics import compute_cer, compute_wer


@dataclass
class RunReport:
    engine: str
    dataset: str
    cer: float
    wer: float
    fps: float
    n_samples: int
    timestamp: str


def _load_labels(tsv: Path) -> list[tuple[str, str]]:
    items = []
    for line in tsv.read_text(encoding="utf-8").splitlines():
        if not line or "\t" not in line:
            continue
        path, text = line.split("\t", 1)
        items.append((path, text))
    return items


def _run_spinai(images_root: Path, items: list[tuple[str, str]]) -> tuple[list[str], float]:
    from spinaiocr.inference.pipeline import OCRPipeline

    pipe = OCRPipeline(lang="ko", tier="lite")
    hyps = []
    start = time.perf_counter()
    for rel_path, _ in items:
        out = pipe(str(images_root / rel_path))
        hyps.append(" ".join(l.text for l in out.lines))
    elapsed = time.perf_counter() - start
    fps = len(items) / max(elapsed, 1e-6)
    return hyps, fps


def _run_paddle(images_root: Path, items: list[tuple[str, str]]) -> tuple[list[str], float]:
    try:
        from paddleocr import PaddleOCR  # type: ignore
    except ImportError:
        raise click.ClickException("paddleocr not installed. pip install paddleocr paddlepaddle")
    pp = PaddleOCR(lang="korean", use_angle_cls=True, show_log=False)
    hyps = []
    start = time.perf_counter()
    for rel_path, _ in items:
        res = pp.ocr(str(images_root / rel_path), cls=True)
        text_parts = []
        for page in res or []:
            for _, (txt, _conf) in page or []:
                text_parts.append(txt)
        hyps.append(" ".join(text_parts))
    elapsed = time.perf_counter() - start
    fps = len(items) / max(elapsed, 1e-6)
    return hyps, fps


_ENGINES = {
    "spinai": _run_spinai,
    "paddle": _run_paddle,
}


@click.command(context_settings={"show_default": True})
@click.option("--images", "images_root", required=True, type=click.Path(exists=True))
@click.option("--labels", required=True, type=click.Path(exists=True), help="TSV: path\\ttext")
@click.option("--engine", default="spinai", type=click.Choice(list(_ENGINES)))
@click.option("--dataset-name", default="custom")
@click.option("--out-dir", default="benchmarks/reports", type=click.Path())
def main(images_root: str, labels: str, engine: str, dataset_name: str, out_dir: str) -> None:
    items = _load_labels(Path(labels))
    refs = [t for _, t in items]

    runner = _ENGINES[engine]
    hyps, fps = runner(Path(images_root), items)

    cer = compute_cer(hyps, refs)
    wer = compute_wer(hyps, refs)
    report = RunReport(
        engine=engine,
        dataset=dataset_name,
        cer=cer.value,
        wer=wer.value,
        fps=fps,
        n_samples=len(items),
        timestamp=datetime.utcnow().isoformat() + "Z",
    )
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset_name}_{engine}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    (out_path / f"{stem}.json").write_text(
        json.dumps(asdict(report), indent=2), encoding="utf-8"
    )
    click.echo(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
