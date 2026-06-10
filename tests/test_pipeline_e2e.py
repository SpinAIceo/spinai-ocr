"""End-to-end pipeline regression tests.

Catches bugs like the ImageNet-normalization mismatch discovered on
2026-04-21 where `OCRPipeline` preprocessing diverged from training's
plain `/255` path, producing hallucinated output ("구도구도구") across
all inputs.

These tests run CPU-only and are slow-ish (~20s total); skip if model
checkpoints are absent so CI without weights still passes.
"""
from __future__ import annotations

import random
from pathlib import Path

import pytest
from PIL import Image

CHECKPOINTS_PRESENT = (Path("checkpoints/lite/ko/rec.pth").exists()
                       and Path("checkpoints/lite/ko/det.pth").exists())

# Quality/accuracy gates must validate the *deployed* tier, not the 14 MB
# lite stub. consumer_v1 is what ships (Railway REC_SHA_PIN); the CER<0.30 /
# BIRTHDAY thresholds below were calibrated against its tuned bench (0.029).
# Running them on lite was a mis-wiring: lite is a weak model (detcrops 0.733)
# and would fail thresholds it was never meant to meet. Skip cleanly when the
# (gitignored) consumer_v1 weights are absent so weightless CI still passes.
PROD_TIER = "consumer_v1"
PROD_PRESENT = (Path(f"checkpoints/{PROD_TIER}/ko/rec.pth").exists()
                and Path(f"checkpoints/{PROD_TIER}/ko/det.pth").exists())

pytestmark = pytest.mark.skipif(
    not CHECKPOINTS_PRESENT, reason="production checkpoints not installed"
)


def _pipeline(tier: str = "lite"):
    from spinaiocr.inference.pipeline import OCRPipeline
    return OCRPipeline(lang="ko", tier=tier, device="cpu")


def test_single_line_auto_detected_and_recognized(tmp_path):
    """A line crop (48 px tall) should trigger single_line path and read correctly.

    Uses subset_ko/00000000.png which we know is 'BIRTHDAY 2026-04-19'.
    """
    if not PROD_PRESENT:
        pytest.skip(f"deployed tier {PROD_TIER!r} weights not installed")
    src = Path("data/synthetic/subset_ko/00000000.png")
    if not src.exists():
        pytest.skip("subset_ko test data not installed")
    pipe = _pipeline(PROD_TIER)
    img = Image.open(src).convert("RGB")
    result = pipe(img)
    assert len(result.lines) == 1, "single-line auto-detect should collapse to one line"
    text = result.lines[0].text
    assert "BIRTHDAY" in text or "2026" in text, f"expected BIRTHDAY/2026 in {text!r}"


def test_detcrops_cer_stays_reasonable():
    """Pipeline CER on detcrops held-out should not regress past 0.3 (vs our
    tuned beam+LM bench of 0.029). Anything dramatically higher indicates
    a preprocessing/normalization regression."""
    if not PROD_PRESENT:
        pytest.skip(f"deployed tier {PROD_TIER!r} weights not installed")
    src = Path("data/detcrops_ko/labels.tsv")
    if not src.exists():
        pytest.skip("detcrops test data not installed")

    from spinaiocr.benchmark.metrics import compute_cer

    pipe = _pipeline(PROD_TIER)
    lines = [ln for ln in src.read_text(encoding="utf-8").splitlines() if "\t" in ln]
    rng = random.Random(42); rng.shuffle(lines)
    sample = lines[:20]  # small — keep test quick

    refs: list[str] = []; hyps: list[str] = []
    for ln in sample:
        path, gt = ln.split("\t", 1)
        full = src.parent / path
        if not full.exists():
            continue
        img = Image.open(full).convert("RGB")
        r = pipe(img)
        # Use first non-empty line (single-line mode returns exactly one)
        hyp = next((l.text for l in r.lines if l.text.strip()), "")
        refs.append(gt); hyps.append(hyp)

    assert refs, "no samples ran"
    cer = compute_cer(hyps, refs).value
    print(f"pipeline detcrops CER ({len(refs)} samples): {cer:.4f}")
    assert cer < 0.30, f"CER {cer:.4f} is too high — likely preprocessing regression"


def test_concurrent_decode_modes_dont_interfere():
    """Two threads, one decoding beam_lm, the other greedy, over many
    iterations. Each must consistently get its own mode's output (no
    cross-thread bleed). Guards against reintroducing a decode_mode
    written to shared pipeline config instead of passed per-call.
    """
    import threading
    import numpy as np
    from PIL import Image
    src = Path("data/synthetic/subset_ko/00000000.png")
    if not src.exists():
        pytest.skip("subset_ko test data not installed")
    pipe = _pipeline()
    img = np.asarray(Image.open(src).convert("RGB"))
    # Ground truth texts per mode (captured by running once)
    expected = {m: pipe(img, decode_mode=m).lines[0].text for m in ("beam_lm", "greedy")}

    results: dict[str, list[str]] = {"beam_lm": [], "greedy": []}
    def worker(mode: str, n: int):
        for _ in range(n):
            r = pipe(img, decode_mode=mode)
            results[mode].append(r.lines[0].text if r.lines else "")

    ta = threading.Thread(target=worker, args=("beam_lm", 10))
    tb = threading.Thread(target=worker, args=("greedy", 10))
    ta.start(); tb.start(); ta.join(); tb.join()
    for mode, got in results.items():
        # Every call must return the expected-for-its-mode text — never the other mode's.
        assert all(g == expected[mode] for g in got), (
            f"{mode} got divergent results under concurrency: "
            f"{set(got)} vs expected {expected[mode]!r}"
        )


def test_tiny_image_returns_empty():
    """Guardrail: images too small for any legible text must produce no
    lines — previously a 1×1 white image returned a hallucinated line
    with confidence ~1.0 because single-line auto-detect fired.
    """
    import numpy as np
    pipe = _pipeline()
    for dim in (1, 4, 7):
        img = np.full((dim, dim, 3), 255, dtype=np.uint8)
        result = pipe(img)
        assert result.lines == [], (
            f"{dim}x{dim} image should produce no lines, got "
            f"{[l.text for l in result.lines]}"
        )


def test_hallucination_output_never_dominates(tmp_path):
    """Guardrail: no single character should make up >70% of any predicted
    line. Catches the "구도구도구" failure mode.
    """
    src = Path("data/synthetic/subset_ko/00000000.png")
    if not src.exists():
        pytest.skip("test image missing")
    pipe = _pipeline()
    result = pipe(Image.open(src).convert("RGB"))
    for line in result.lines:
        s = line.text.strip()
        if len(s) < 4:
            continue
        most_common = max(set(s), key=s.count)
        frac = s.count(most_common) / len(s)
        assert frac < 0.70, (
            f"predicted text {s!r} has {most_common!r} at {frac:.0%} "
            f"— signature of hallucinated output"
        )
