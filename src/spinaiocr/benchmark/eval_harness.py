"""Held-out evaluation harness with regression gates.

`eval_recognizer()` runs a recognition checkpoint over a fixed eval split
and returns CER/WER plus per-bucket breakdowns. A baseline JSON (last
known-good CER) is stored per dataset; if the new CER > baseline ×
`regression_margin`, the gate raises `RegressionError`.

Typical usage in CI::

    from spinaiocr.benchmark.eval_harness import RegressionGate, eval_recognizer

    metrics = eval_recognizer(
        ckpt_path="checkpoints/lite/ko/rec.pth",
        eval_root="data/eval/ko_v1",
        eval_labels="data/eval/ko_v1/labels.tsv",
    )
    RegressionGate("ko_v1").check(metrics)   # raises if worse than baseline
"""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from spinaiocr.benchmark.metrics import _levenshtein_seq, compute_cer, compute_wer
from spinaiocr.io import save_json
from spinaiocr.log import get_logger, log_span
from spinaiocr.models.recognition import build_recognition
from spinaiocr.training.datamodule import RecognitionTrainingDataset, recognition_collate
from spinaiocr.vocab.base import Vocab, load_vocab

log = get_logger("spinaiocr.benchmark.eval")


class RegressionError(AssertionError):
    pass


@dataclass
class EvalMetrics:
    dataset: str
    cer: float
    wer: float
    n_samples: int
    samples_by_length_bucket: dict = field(default_factory=dict)
    confused_pairs: list = field(default_factory=list)
    elapsed_s: float = 0.0
    ckpt_path: str = ""

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "cer": self.cer,
            "wer": self.wer,
            "n_samples": self.n_samples,
            "samples_by_length_bucket": self.samples_by_length_bucket,
            "confused_pairs": self.confused_pairs[:50],
            "elapsed_s": self.elapsed_s,
            "ckpt_path": self.ckpt_path,
        }


def _length_bucket(n: int) -> str:
    if n <= 5:
        return "short(≤5)"
    if n <= 15:
        return "medium(6-15)"
    if n <= 40:
        return "long(16-40)"
    return "verylong(>40)"


def _confused_char_pairs(hyps: list[str], refs: list[str], top: int = 50) -> list:
    pairs: Counter = Counter()
    for h, r in zip(hyps, refs):
        # align via edit ops would be ideal; use char-index alignment as
        # an approximation by min-length comparison (cheap).
        for a, b in zip(h, r):
            if a != b:
                pairs[(b, a)] += 1
    return [{"ref": b, "hyp": a, "count": c} for (b, a), c in pairs.most_common(top)]


@torch.no_grad()
def _decode_dataset(
    model, loader: DataLoader, vocab: Vocab, device: str
) -> list[str]:
    hyps: list[str] = []
    for batch in loader:
        logits = model(batch["image"].to(device))
        for row in logits.argmax(-1).cpu().numpy():
            hyps.append(vocab.decode(row, ctc_collapse=True))
    return hyps


def eval_recognizer(
    ckpt_path: str | Path,
    eval_root: str | Path,
    eval_labels: str | Path,
    *,
    device: str = "auto",
    batch_size: int = 64,
    dataset_name: str = "unnamed_eval",
    save_report_to: str | Path | None = None,
) -> EvalMetrics:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    started = time.perf_counter()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = ckpt.get("arch", "svtr_lite")

    # vocab: prefer chars stored in ckpt, else named vocab
    vocab_name = ckpt.get("vocab_name", "ko_en_v1")
    if "vocab_chars" in ckpt:
        from spinaiocr.vocab.base import JamoVocab
        cls = JamoVocab if vocab_name == "jamo_ko_v1" else Vocab
        vocab = cls(name=vocab_name, chars=ckpt["vocab_chars"])
    else:
        vocab = load_vocab(vocab_name)
    model = build_recognition(arch, vocab_size=vocab.size,
                              input_height=ckpt.get("input_height", 48)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    ds = RecognitionTrainingDataset(
        Path(eval_root), Path(eval_labels), vocab,
        image_height=ckpt.get("input_height", 48), max_width=320,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=recognition_collate, num_workers=0)

    with log_span("eval.decode", n=len(ds), ckpt=str(ckpt_path)):
        hyps = _decode_dataset(model, loader, vocab, device)
    refs = [text for _, text in ds.base.items]
    cer = compute_cer(hyps, refs)
    wer = compute_wer(hyps, refs)

    # Length buckets
    buckets: dict = defaultdict(lambda: {"n": 0, "edits": 0, "chars": 0})
    for h, r in zip(hyps, refs):
        b = _length_bucket(len(r))
        buckets[b]["n"] += 1
        buckets[b]["edits"] += _levenshtein_seq(h, r)
        buckets[b]["chars"] += max(len(r), 1)
    for b in buckets:
        buckets[b]["cer"] = buckets[b]["edits"] / max(buckets[b]["chars"], 1)

    confused = _confused_char_pairs(hyps, refs)
    elapsed = time.perf_counter() - started

    metrics = EvalMetrics(
        dataset=dataset_name, cer=cer.value, wer=wer.value, n_samples=cer.n,
        samples_by_length_bucket=dict(buckets), confused_pairs=confused,
        elapsed_s=elapsed, ckpt_path=str(ckpt_path),
    )
    log.info(
        "eval.result dataset=%s ckpt=%s cer=%.4f wer=%.4f n=%d elapsed_s=%.1f",
        dataset_name, ckpt_path, cer.value, wer.value, cer.n, elapsed,
        extra={"op": "eval.result", "dataset": dataset_name,
               "cer": cer.value, "wer": wer.value, "n": cer.n},
    )
    if save_report_to:
        save_json(save_report_to, metrics.to_dict())
    return metrics


# ---------------------------------------------------------------------------
# Regression gate
# ---------------------------------------------------------------------------


@dataclass
class RegressionGate:
    dataset: str
    baseline_path: Path = Path("benchmarks/baselines")
    regression_margin: float = 1.05  # allow up to 5% worse before FAIL

    @property
    def _baseline_file(self) -> Path:
        return self.baseline_path / f"{self.dataset}.json"

    def read_baseline(self) -> dict | None:
        p = self._baseline_file
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def check(self, metrics: EvalMetrics) -> None:
        baseline = self.read_baseline()
        if baseline is None:
            log.warning("gate.no_baseline dataset=%s — will install current as baseline",
                        self.dataset,
                        extra={"op": "regression.check", "dataset": self.dataset})
            self.install(metrics)
            return
        prev_cer = float(baseline.get("cer", 1.0))
        if metrics.cer > prev_cer * self.regression_margin:
            log.error(
                "gate.regression dataset=%s baseline_cer=%.4f new_cer=%.4f margin=%.2fx",
                self.dataset, prev_cer, metrics.cer, self.regression_margin,
                extra={"op": "regression.fail", "dataset": self.dataset,
                       "baseline_cer": prev_cer, "new_cer": metrics.cer},
            )
            raise RegressionError(
                f"CER on {self.dataset} regressed: {prev_cer:.4f} → {metrics.cer:.4f} "
                f"(margin {self.regression_margin:.2f}x)"
            )
        log.info(
            "gate.pass dataset=%s baseline=%.4f new=%.4f delta=%+.4f",
            self.dataset, prev_cer, metrics.cer, metrics.cer - prev_cer,
            extra={"op": "regression.pass", "dataset": self.dataset,
                   "baseline_cer": prev_cer, "new_cer": metrics.cer},
        )

    def install(self, metrics: EvalMetrics) -> None:
        """Promote current metrics as the new baseline."""
        save_json(self._baseline_file, metrics.to_dict())
        log.info("gate.baseline_installed dataset=%s cer=%.4f", self.dataset, metrics.cer,
                 extra={"op": "regression.install", "dataset": self.dataset,
                        "cer": metrics.cer})
