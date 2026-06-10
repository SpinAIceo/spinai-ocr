"""Tests for experiment tracking, eval harness, error analysis, decoders,
curriculum, TTA, and model registry."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch


def test_experiment_tracker_writes_all(tmp_path, monkeypatch):
    from spinai_ocr.experiments import ExperimentTracker

    monkeypatch.setenv("SPINAI_LOG_DIR", str(tmp_path / "logs"))
    tracker = ExperimentTracker.create(
        "unit_test", config={"lr": 1e-3}, root=str(tmp_path / "exp"),
        use_wandb=False,
    )
    tracker.log_metrics(step=0, loss=5.0, cer=0.9)
    tracker.log_metrics(step=100, loss=1.0, cer=0.15)
    tracker.log_artifact(tmp_path / "dummy.pth", kind="checkpoint")
    tracker.finish(final_metrics={"best_cer": 0.12})

    run_dir = tracker.run_dir
    assert (run_dir / "config.json").exists()
    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "artifacts.jsonl").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "README.md").exists()

    import json
    metrics = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(metrics) == 2
    assert metrics[1]["cer"] == 0.15


def test_greedy_and_beam_decoders_agree_on_trivial():
    """On a near-deterministic output, greedy and beam should pick the same path."""
    from spinai_ocr.inference.decoders import BeamConfig, ctc_beam_search, ctc_greedy
    from spinai_ocr.vocab.base import Vocab

    v = Vocab(name="tiny", chars=list("abc"))
    T = 6
    # Craft a sequence that cleanly decodes to "abc" under CTC:
    # blank, a, a, blank, b, c  → "abc"
    logits = np.full((T, v.size), -10.0, dtype=np.float32)
    script = [v.blank_id, v._ctoi["a"], v._ctoi["a"], v.blank_id, v._ctoi["b"], v._ctoi["c"]]
    for t, cid in enumerate(script):
        logits[t, cid] = 10.0
    greedy = ctc_greedy(logits, v)
    beam = ctc_beam_search(logits, v, cfg=BeamConfig(beam_width=4))
    assert greedy == "abc"
    assert beam == "abc"


def test_char_bigram_lm_scores_known_transitions():
    from spinai_ocr.inference.decoders import CharBigramLM

    lm = CharBigramLM(alpha=0.5)
    lm.fit(["hello", "world", "help"])
    # p('e' after 'h') should be higher than p('z' after 'h')
    assert lm.logprob("h", "e") > lm.logprob("h", "z")


def test_error_analysis_report_basic():
    from spinai_ocr.benchmark.error_analysis import analyze

    hyps = ["hello", "wold", "help"]
    refs = ["hello", "world", "help"]
    rep = analyze(hyps, refs)
    assert rep.n_pairs == 3
    assert rep.cer > 0
    assert any(c["ref"] == "r" for c in rep.top_sub_pairs) or any(c["char"] == "r" for c in rep.top_del_chars)


def test_curriculum_sampler_expands_over_time():
    from spinai_ocr.training.curriculum import (
        CurriculumConfig, CurriculumSampler,
    )

    difficulty = np.arange(100).astype(np.float32)  # 0 easiest, 99 hardest
    sampler = CurriculumSampler(
        difficulty, CurriculumConfig(total_steps=1000, start_frac=0.1, warmup_frac=0.5),
        seed=0,
    )
    sampler.set_step(0)
    it = iter(sampler)
    first_samples = [next(it) for _ in range(20)]
    # at step 0 the window is first 10 indices
    assert max(first_samples) < 20

    sampler.set_step(800)  # past warmup
    late_samples = [next(it) for _ in range(200)]
    assert max(late_samples) >= 80  # full range exposed


def test_tta_predict_shape_matches():
    from spinai_ocr.inference.tta_ensemble import TTAConfig, tta_predict

    class Tiny(torch.nn.Module):
        def forward(self, x):
            b = x.size(0)
            # pretend output [B, T=4, V=5]
            return torch.randn(b, 4, 5, device=x.device)

    x = torch.randn(2, 3, 48, 96)
    out = tta_predict(Tiny(), x, TTAConfig(scales=(1.0, 1.1), rotations_deg=(0.0, 1.0)))
    assert out.shape == (2, 4, 5)
    # softmax → rows sum to 1
    assert torch.allclose(out.sum(-1), torch.ones_like(out.sum(-1)), atol=1e-4)


def test_model_registry_register_and_best(tmp_path):
    from spinai_ocr.models.registry import ModelRegistry

    # create a fake ckpt
    ckpt = tmp_path / "fake.pth"
    torch.save({"state_dict": {"a": torch.zeros(1)}}, ckpt)

    reg = ModelRegistry(root=tmp_path / "registry")
    e1 = reg.register(kind="recognition", lang="ko", tag="svtr",
                      ckpt_path=ckpt, metrics={"cer": 0.2})
    e2 = reg.register(kind="recognition", lang="ko", tag="svtr",
                      ckpt_path=ckpt, metrics={"cer": 0.1})
    assert e1.version == 1 and e2.version == 2
    best = reg.best("recognition", lang="ko", metric="cer")
    assert best is not None and best.metrics["cer"] == 0.1


# iter 140: promote_to_current pre-fix hardcoded tier="lite". Promoting a
# consumer_v1 entry silently overwrote `checkpoints/lite/ko/rec.pth`
# instead of `checkpoints/consumer_v1/ko/rec.pth`. Pin all three paths.

def test_promote_to_current_explicit_tier(tmp_path):
    from spinai_ocr.models.registry import ModelRegistry

    ckpt = tmp_path / "rec.pth"
    torch.save({"state_dict": {"a": torch.zeros(1)}}, ckpt)
    reg = ModelRegistry(root=tmp_path / "registry")
    entry = reg.register(kind="recognition", lang="ko", tag="svtr_deep",
                         ckpt_path=ckpt, metrics={"cer": 0.1})
    out_root = tmp_path / "out"
    dest = reg.promote_to_current(entry, checkpoints_root=out_root,
                                   tier="consumer_v1")
    assert dest == out_root / "consumer_v1" / "ko" / "rec.pth"
    assert dest.exists()
    # crucially: lite/ should NOT have been touched
    assert not (out_root / "lite" / "ko" / "rec.pth").exists()


def test_promote_to_current_tier_from_config(tmp_path):
    from spinai_ocr.models.registry import ModelRegistry

    ckpt = tmp_path / "rec.pth"
    torch.save({"state_dict": {"a": torch.zeros(1)}}, ckpt)
    reg = ModelRegistry(root=tmp_path / "registry")
    entry = reg.register(kind="recognition", lang="ko", tag="svtr_deep",
                         ckpt_path=ckpt, metrics={"cer": 0.1},
                         config={"arch": "svtr_deep", "tier": "medical"})
    out_root = tmp_path / "out"
    dest = reg.promote_to_current(entry, checkpoints_root=out_root)
    assert dest == out_root / "medical" / "ko" / "rec.pth"
    assert dest.exists()


def test_promote_to_current_defaults_to_lite_for_back_compat(tmp_path):
    """Entries registered before iter 140 don't carry a 'tier' in config.
    For those, default to 'lite' so old test fixtures + scripts keep
    working byte-identical."""
    from spinai_ocr.models.registry import ModelRegistry

    ckpt = tmp_path / "rec.pth"
    torch.save({"state_dict": {"a": torch.zeros(1)}}, ckpt)
    reg = ModelRegistry(root=tmp_path / "registry")
    entry = reg.register(kind="recognition", lang="ko", tag="svtr_lite",
                         ckpt_path=ckpt, metrics={"cer": 0.1})
    out_root = tmp_path / "out"
    dest = reg.promote_to_current(entry, checkpoints_root=out_root)
    assert dest == out_root / "lite" / "ko" / "rec.pth"
    assert dest.exists()


def test_regression_gate_installs_baseline(tmp_path):
    from spinai_ocr.benchmark.eval_harness import (
        EvalMetrics, RegressionGate,
    )

    gate = RegressionGate(dataset="toy", baseline_path=tmp_path / "b")
    metrics = EvalMetrics(dataset="toy", cer=0.1, wer=0.2, n_samples=100, elapsed_s=1.0)
    gate.check(metrics)  # installs baseline (no prior)
    # second run with same cer must pass
    gate.check(metrics)
    # run that regresses beyond margin must fail
    bad = EvalMetrics(dataset="toy", cer=0.2, wer=0.3, n_samples=100)
    with pytest.raises(Exception):
        gate.check(bad)


def test_logged_io_atomic_write_bytes(tmp_path):
    from spinai_ocr.io import atomic_write_bytes, sha256_of_file

    path = tmp_path / "sub" / "file.bin"
    atomic_write_bytes(path, b"hello world", sha256=False)
    assert path.read_bytes() == b"hello world"
    # sha256 matches
    expected = sha256_of_file(path)
    assert len(expected) == 64


def test_logged_io_save_jsonl_append(tmp_path):
    from spinai_ocr.io import save_jsonl_append

    path = tmp_path / "x.jsonl"
    n = save_jsonl_append(path, [{"a": 1}, {"b": 2}])
    assert n == 2
    content = path.read_text(encoding="utf-8").splitlines()
    assert len(content) == 2
