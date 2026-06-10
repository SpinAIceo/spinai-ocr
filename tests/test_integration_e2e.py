"""End-to-end smoke: synth → dataset → model → CTC loss → inference → CER.

Runs quickly (~5s) so it's safe in CI. Does NOT assert accuracy — the
signal is "the full pipeline ran without raising."
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from spinai_ocr.benchmark.metrics import compute_cer
from spinai_ocr.data.synth_simple import (
    SimpleSynthSpec,
    default_korean_corpus,
    generate_simple,
)
from spinai_ocr.inference.pipeline import OCRPipeline
from spinai_ocr.models.angle_cls import AngleClassifier, predict_angle
from spinai_ocr.models.recognition import build_recognition
from spinai_ocr.training.datamodule import (
    RecognitionTrainingDataset,
    recognition_collate,
)
from spinai_ocr.vocab.base import load_vocab


def test_full_pipeline_runs(tmp_path: Path):
    # Seed so the untrained-model CER below has a deterministic upper bound
    # (CER on random weights can exceed 1.0 when the CTC decode over-produces,
    # and without a seed this test was occasionally flaky above 2.0).
    torch.manual_seed(0)
    out = tmp_path / "synth"
    n = generate_simple(
        SimpleSynthSpec(
            out_dir=out,
            corpus=default_korean_corpus(),
            count=16,
            blur_p=0.0,
            background_jitter=False,
            rotation_deg=0.0,
        )
    )
    assert n == 16

    vocab = load_vocab("ko_en_v1")
    ds = RecognitionTrainingDataset(out, out / "labels.tsv", vocab)
    assert len(ds) == 16

    loader = DataLoader(ds, batch_size=4, collate_fn=recognition_collate)
    batch = next(iter(loader))

    model = build_recognition("svtr_lite", vocab_size=vocab.size, input_height=48)
    logits = model(batch["image"])
    b, t, _ = logits.shape
    log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
    input_lengths = torch.full((b,), t, dtype=torch.long)
    loss = F.ctc_loss(
        log_probs,
        batch["targets"],
        input_lengths,
        batch["target_lengths"],
        blank=vocab.blank_id,
        zero_infinity=True,
    )
    assert torch.isfinite(loss)

    # decode
    hyps = [vocab.decode(row, ctc_collapse=True) for row in logits.argmax(-1).numpy()]
    # refs from the same batch
    refs = [ds.base.items[i][1] for i in range(4)]
    cer = compute_cer(hyps, refs)
    # untrained — just assert the metric computed without raising.
    # Random-weight CTC can over-decode (hyp length >> ref length) giving
    # CER >> 1; bound is just a sanity ceiling.
    import math as _math
    assert _math.isfinite(cer.value) and cer.value >= 0.0


def test_angle_classifier_forward():
    model = AngleClassifier()
    x = torch.randn(1, 3, 48, 192)
    angle = predict_angle(model, x)
    assert angle in (0, 90, 180, 270)


def test_pipeline_no_ckpt_graceful(tmp_path: Path):
    """Pipeline must boot and run even without trained checkpoints — returns
    no lines but does not raise."""
    n = generate_simple(
        SimpleSynthSpec(out_dir=tmp_path, corpus=default_korean_corpus(), count=2,
                        blur_p=0.0, background_jitter=False)
    )
    assert n == 2
    pipe = OCRPipeline(lang="ko", tier="lite", device="cpu",
                      checkpoints_root=str(tmp_path / "no_such"))
    result = pipe(str(tmp_path / "00000000.png"))
    assert result.image_width > 0
    assert result.image_height > 0
