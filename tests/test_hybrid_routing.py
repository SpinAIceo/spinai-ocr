"""iter 131: hybrid routing productionization smoke tests.

Exercises the three-mode router (`routing_mode={fast,balanced,accurate}`)
without requiring EasyOCR installed: `_easyocr_recognize_line_crop` is
monkey-patched so the test verifies routing logic only. The real EasyOCR
path is covered indirectly by the iter 128/129/130 bench scripts.

Verifies:
  - Config field validation (routing_mode literal, threshold floats).
  - `_resolve_routing_params` returns mode-specific presets and honors
    user overrides.
  - `_apply_hybrid_routing` is a no-op when routing_mode is None.
  - Per-line replace fires only on lines below `line_thresh` and keeps
    polygons/length unchanged.
  - Whole-image fallback (balanced/accurate) collapses output to a single
    full-image bbox when coverage is below `cov_thresh`.
  - EasyOCR import failure (poison sentinel) does not crash the pipeline.
"""
from __future__ import annotations

import numpy as np
import pytest

from spinaiocr.config import PipelineConfig, RecognitionConfig
from spinaiocr.inference.pipeline import OCRPipeline


def _bare_pipeline(routing_mode=None, **rec_kwargs):
    """Build an OCRPipeline without loading checkpoints — we only test
    routing helpers, not detection/recognition. ``routing_mode`` may be
    None (legacy) or one of the three preset names."""
    cfg = PipelineConfig(
        lang="ko",
        tier="lite",
        device="cpu",
        recognition=RecognitionConfig(routing_mode=routing_mode, **rec_kwargs),
    )
    return OCRPipeline(config=cfg, checkpoints_root="/__nonexistent__")


def test_config_accepts_routing_mode_literal():
    for mode in ("fast", "balanced", "accurate", None):
        cfg = RecognitionConfig(routing_mode=mode)
        assert cfg.routing_mode == mode
    with pytest.raises(Exception):
        RecognitionConfig(routing_mode="bogus")


def test_resolve_routing_params_presets():
    pipe_fast = _bare_pipeline("fast")
    assert pipe_fast._resolve_routing_params() == ("fast", 0.70, None, "min_conf")

    pipe_bal = _bare_pipeline("balanced")
    assert pipe_bal._resolve_routing_params() == ("balanced", 0.70, 0.65, "min_conf")

    pipe_acc = _bare_pipeline("accurate")
    assert pipe_acc._resolve_routing_params() == ("accurate", 0.90, 0.80, "min_conf")

    pipe_none = _bare_pipeline(None)
    assert pipe_none._resolve_routing_params() is None


def test_resolve_routing_params_user_overrides():
    pipe = _bare_pipeline(
        "balanced",
        routing_line_thresh=0.55,
        routing_cov_thresh=0.40,
        routing_cov_signal="mean_conf",
    )
    assert pipe._resolve_routing_params() == ("balanced", 0.55, 0.40, "mean_conf")


def test_apply_routing_noop_when_disabled():
    pipe = _bare_pipeline(None)
    img = np.zeros((48, 200, 3), dtype=np.uint8)
    polys = [np.array([[10, 10], [100, 10], [100, 40], [10, 40]], dtype=np.float32)]
    texts = ["hi"]
    conf = np.array([0.30], dtype=np.float32)
    mcc = np.array([0.20], dtype=np.float32)
    p2, t2, c2, m2 = pipe._apply_hybrid_routing(img, polys, texts, conf, mcc)
    assert (p2, t2) == (polys, texts)
    assert np.array_equal(c2, conf) and np.array_equal(m2, mcc)


def test_per_line_replace_only_below_threshold(monkeypatch):
    pipe = _bare_pipeline("fast")  # line_thresh=0.70, no cov gate

    def fake_recognize(self, img, polygon, **_):
        return f"easy_{int(polygon[0][0])}"

    monkeypatch.setattr(OCRPipeline, "_easyocr_recognize_line_crop",
                        fake_recognize, raising=False)

    img = np.zeros((48, 400, 3), dtype=np.uint8)
    polys = [
        np.array([[0, 10], [80, 10], [80, 40], [0, 40]], dtype=np.float32),
        np.array([[100, 10], [180, 10], [180, 40], [100, 40]], dtype=np.float32),
        np.array([[200, 10], [280, 10], [280, 40], [200, 40]], dtype=np.float32),
    ]
    texts = ["lo", "hi", "mid"]
    conf = np.array([0.30, 0.95, 0.69], dtype=np.float32)
    mcc = np.array([0.10, 0.50, 0.40], dtype=np.float32)

    p2, t2, c2, m2 = pipe._apply_hybrid_routing(img, polys, texts, conf, mcc)
    assert p2 is polys, "per-line route must keep polygons unchanged"
    assert t2 == ["easy_0", "hi", "easy_200"], f"unexpected: {t2}"
    assert c2[0] == 1.0 and c2[2] == 1.0
    assert np.isclose(c2[1], 0.95)
    # min_char_confs is unmodified by per-line replace
    assert np.array_equal(m2, mcc)


def test_whole_image_fallback_triggers_below_cov(monkeypatch):
    pipe = _bare_pipeline("balanced")  # cov=min_conf<0.65 → whole-image
    monkeypatch.setattr(OCRPipeline, "_easyocr_fallback",
                        lambda self, img: "WHOLE_IMAGE_TEXT", raising=False)
    # Patch per-line so it would change things if it ran — proving we took
    # the whole-image branch and never invoked per-line.
    monkeypatch.setattr(OCRPipeline, "_easyocr_recognize_line_crop",
                        lambda *a, **kw: "PER_LINE_SHOULD_NOT_FIRE", raising=False)

    img = np.zeros((100, 300, 3), dtype=np.uint8)
    polys = [
        np.array([[0, 10], [80, 10], [80, 40], [0, 40]], dtype=np.float32),
        np.array([[100, 10], [180, 10], [180, 40], [100, 40]], dtype=np.float32),
    ]
    texts = ["lo", "ok"]
    conf = np.array([0.40, 0.99], dtype=np.float32)  # min=0.40 < 0.65
    mcc = np.array([0.10, 0.50], dtype=np.float32)

    p2, t2, c2, m2 = pipe._apply_hybrid_routing(img, polys, texts, conf, mcc)
    assert len(p2) == 1 and len(t2) == 1
    assert t2 == ["WHOLE_IMAGE_TEXT"]
    # full-image bbox: 300×100
    poly_full = p2[0]
    assert poly_full.shape == (4, 2)
    assert poly_full[0].tolist() == [0.0, 0.0]
    assert poly_full[2].tolist() == [300.0, 100.0]


def test_whole_image_fallback_skipped_when_cov_above(monkeypatch):
    pipe = _bare_pipeline("balanced")
    monkeypatch.setattr(OCRPipeline, "_easyocr_fallback",
                        lambda self, img: "SHOULD_NOT_FIRE", raising=False)
    monkeypatch.setattr(OCRPipeline, "_easyocr_recognize_line_crop",
                        lambda *a, **kw: None, raising=False)
    img = np.zeros((100, 300, 3), dtype=np.uint8)
    polys = [np.array([[0, 0], [80, 0], [80, 40], [0, 40]], dtype=np.float32)]
    texts = ["clean"]
    conf = np.array([0.99], dtype=np.float32)  # above cov 0.65 AND above line 0.70
    mcc = np.array([0.50], dtype=np.float32)
    p2, t2, c2, m2 = pipe._apply_hybrid_routing(img, polys, texts, conf, mcc)
    assert t2 == ["clean"], "high-conf input should not be routed"
    assert len(p2) == 1


def test_easyocr_import_failure_is_silent(monkeypatch):
    """If EasyOCR is missing/broken, routing must degrade silently — return
    inputs unchanged rather than raising."""
    pipe = _bare_pipeline("fast")
    pipe._easyocr_reader = False  # poison sentinel

    img = np.zeros((48, 200, 3), dtype=np.uint8)
    polys = [np.array([[10, 10], [100, 10], [100, 40], [10, 40]], dtype=np.float32)]
    texts = ["lowconf"]
    conf = np.array([0.30], dtype=np.float32)
    mcc = np.array([0.10], dtype=np.float32)
    p2, t2, c2, m2 = pipe._apply_hybrid_routing(img, polys, texts, conf, mcc)
    assert t2 == ["lowconf"], "no EasyOCR → input passed through unchanged"
    assert np.array_equal(c2, conf)


def test_zero_lines_with_cov_gate_falls_back(monkeypatch):
    """Whole-image fallback must trigger even when v031 produced zero lines
    (matches iter130 'whole_zero' mode in the bench)."""
    pipe = _bare_pipeline("balanced")
    monkeypatch.setattr(OCRPipeline, "_easyocr_fallback",
                        lambda self, img: "FALLBACK_TEXT", raising=False)
    img = np.zeros((100, 300, 3), dtype=np.uint8)
    p2, t2, c2, m2 = pipe._apply_hybrid_routing(img, [], [], np.array([]), np.array([]))
    assert t2 == ["FALLBACK_TEXT"]
    assert len(p2) == 1


# iter 132: single-line input parity. Same _apply_hybrid_routing helper, but
# called with a 1-element input list shaped like the single_line branch in
# OCRPipeline.__call__ produces (full-image polygon, single conf/mcc).

def _single_line_inputs(W=832, H=59, conf=0.69, text="partial v031"):
    img = np.zeros((H, W, 3), dtype=np.uint8)
    poly = np.array([[0.0, 0.0], [float(W), 0.0],
                     [float(W), float(H)], [0.0, float(H)]], dtype=np.float32)
    return img, [poly], [text], np.array([conf], np.float32), np.array([conf], np.float32)


def test_single_line_per_line_replace_below_threshold(monkeypatch):
    """iter132: single_line branch with routing_mode=fast should per-line-replace
    when v031 conf < line_thresh. Mirrors the live smoke iter132 result on
    det_ko_v2/00000003.png (conf 0.69 → replaced, conf becomes 1.0)."""
    pipe = _bare_pipeline("fast")  # line=0.70, no cov gate
    monkeypatch.setattr(OCRPipeline, "_easyocr_recognize_line_crop",
                        lambda self, img, poly, **kw: "EASYOCR_FULL_LINE",
                        raising=False)
    img, polys, texts, conf, mcc = _single_line_inputs(conf=0.69)
    p2, t2, c2, m2 = pipe._apply_hybrid_routing(img, polys, texts, conf, mcc)
    assert t2 == ["EASYOCR_FULL_LINE"], "single_line per-line replace must fire"
    assert c2[0] == 1.0
    # polygon (full-image bbox) preserved
    assert len(p2) == 1 and p2[0].shape == (4, 2)


def test_single_line_whole_image_fallback_fires(monkeypatch):
    """iter132: routing_mode=accurate on a single_line input with conf 0.69 <
    cov_thresh 0.80 → whole-image readtext fallback. Mirrors smoke 'accurate'
    output (conf stays at cov_value = 0.69)."""
    pipe = _bare_pipeline("accurate")  # line=0.90, cov=0.80
    fb_calls = []

    def fake_fb(self, img):
        fb_calls.append(img.shape)
        return "WHOLE_IMAGE_LINE"

    monkeypatch.setattr(OCRPipeline, "_easyocr_fallback", fake_fb, raising=False)
    monkeypatch.setattr(OCRPipeline, "_easyocr_recognize_line_crop",
                        lambda *a, **kw: "PER_LINE_SHOULD_NOT_FIRE",
                        raising=False)
    img, polys, texts, conf, mcc = _single_line_inputs(conf=0.69)
    p2, t2, c2, m2 = pipe._apply_hybrid_routing(img, polys, texts, conf, mcc)
    assert len(t2) == 1 and t2 == ["WHOLE_IMAGE_LINE"]
    assert len(fb_calls) == 1, "whole-image fallback must run exactly once"
    # conf becomes cov_value (min line conf)
    assert np.isclose(c2[0], 0.69)


def test_single_line_high_conf_passthrough(monkeypatch):
    """iter132: high-conf single_line input (conf 0.95 above all thresholds)
    must pass through unchanged in any mode."""
    monkeypatch.setattr(OCRPipeline, "_easyocr_fallback",
                        lambda self, img: "SHOULD_NOT_FIRE", raising=False)
    monkeypatch.setattr(OCRPipeline, "_easyocr_recognize_line_crop",
                        lambda *a, **kw: "SHOULD_NOT_FIRE", raising=False)
    for mode in ("fast", "balanced", "accurate"):
        pipe = _bare_pipeline(mode)
        img, polys, texts, conf, mcc = _single_line_inputs(conf=0.95, text="clean")
        p2, t2, c2, m2 = pipe._apply_hybrid_routing(img, polys, texts, conf, mcc)
        assert t2 == ["clean"], f"mode {mode} must passthrough high-conf input"


def test_easyocr_fallback_converts_rgb_to_bgr(monkeypatch):
    """iter134: `_easyocr_fallback` must convert RGB→BGR before calling
    EasyOCR's readtext. Our internal img is RGB (PIL convention) but
    EasyOCR's reformat_input treats a 3-channel ndarray as BGR. Check that
    the array fed to readtext has its R/B swapped relative to the input.
    """
    pipe = _bare_pipeline(None)
    pipe._easyocr_reader = object()  # bypass lazy init

    captured: list[np.ndarray] = []

    def fake_readtext(self_or_arr, *args, **kwargs):  # noqa: ANN001
        # The reader's bound method receives `self` as the Reader; here we
        # mock as a plain callable, so the first positional is the array.
        captured.append(self_or_arr)
        return []

    class _FakeReader:
        def readtext(self, arr, *a, **kw):  # noqa: D401
            captured.append(arr)
            return []

    pipe._easyocr_reader = _FakeReader()
    rgb = np.zeros((20, 30, 3), dtype=np.uint8)
    rgb[..., 0] = 255  # red-only RGB
    out = pipe._easyocr_fallback(rgb)
    assert out == ""
    assert len(captured) == 1
    fed = captured[0]
    # After RGB→BGR conversion, red channel (idx 0 in RGB) lands in idx 2.
    assert fed.shape == rgb.shape
    assert fed[0, 0, 0] == 0   # B channel of converted BGR
    assert fed[0, 0, 2] == 255 # R channel of converted BGR


def test_easyocr_fallback_passes_grayscale_unchanged(monkeypatch):
    """iter134: 2D grayscale input must NOT be cv2.cvtColor'd."""
    pipe = _bare_pipeline(None)
    captured: list[np.ndarray] = []

    class _FakeReader:
        def readtext(self, arr, *a, **kw):  # noqa: D401
            captured.append(arr)
            return []

    pipe._easyocr_reader = _FakeReader()
    gray = np.full((20, 30), 128, dtype=np.uint8)
    pipe._easyocr_fallback(gray)
    assert len(captured) == 1
    assert captured[0].ndim == 2
    assert np.all(captured[0] == 128)


# iter 141: easyocr_fallback_threshold REMOVED. consumer_v1 default tier
# behavior migrates from legacy single-threshold whole-line fallback to
# routing_mode="balanced" (the iter 130 OOD-Pareto-dominant preset).
# Pin the migration so a future refactor cannot silently disable it.

def test_consumer_v1_tier_default_routing_mode_accurate():
    """iter 144: consumer_v1 default escalated `balanced` → `accurate`.
    iter 143 honest bench on real Korean photos (n=171) found `balanced`
    was a no-op (CER same as None) because v031 conf is mis-calibrated
    on real photos; `accurate` ships -45.4% rel CER (47.80% → 26.11%)
    at p50 +20 ms. Pin the new default."""
    cfg = PipelineConfig(
        lang="ko", tier="consumer_v1", device="cpu",
        recognition=RecognitionConfig(),
    )
    pipe = OCRPipeline(config=cfg, checkpoints_root="/__nonexistent__")
    assert pipe.config.recognition.routing_mode == "accurate"


def test_lite_tier_default_routing_mode_none():
    """lite tier never had a fallback default — must remain None
    (untested tier, backward-compatible behavior)."""
    cfg = PipelineConfig(
        lang="ko", tier="lite", device="cpu",
        recognition=RecognitionConfig(),
    )
    pipe = OCRPipeline(config=cfg, checkpoints_root="/__nonexistent__")
    assert pipe.config.recognition.routing_mode is None


def test_explicit_routing_mode_overrides_tier_default():
    """A caller passing routing_mode='fast' on consumer_v1 keeps 'fast';
    the auto-set must NOT silently overwrite an explicit user value.
    iter 144: changed canary from 'accurate' (now the new default) to
    'fast' so this test still proves override-vs-default rather than
    accidentally testing the default."""
    cfg = PipelineConfig(
        lang="ko", tier="consumer_v1", device="cpu",
        recognition=RecognitionConfig(routing_mode="fast"),
    )
    pipe = OCRPipeline(config=cfg, checkpoints_root="/__nonexistent__")
    assert pipe.config.recognition.routing_mode == "fast"


def test_easyocr_fallback_threshold_field_removed():
    """iter 141 hard-removed `easyocr_fallback_threshold` from the schema.
    Pydantic v2 with default config silently ignores extra fields, so
    passing the legacy keyword is not an error — but the value must NOT
    be retained on the resulting config (no shadow attribute, no behavior
    fork in the pipeline). This pins both: schema attribute is gone, and
    nothing reads it."""
    cfg = RecognitionConfig(easyocr_fallback_threshold=0.60)
    assert not hasattr(cfg, "easyocr_fallback_threshold")
    # Round-trip via model_dump to confirm the value is dropped — a future
    # caller that re-loads from yaml/env must not see it leak back in.
    assert "easyocr_fallback_threshold" not in cfg.model_dump()


# iter 142: schema dead-field sweep. Pin that the cleanup stays clean —
# a future hand re-introducing scaffolding-without-implementation must trip
# these tests, not slip through.

def test_iter142_recognition_dead_fields_stay_removed():
    """`max_input_width` and `max_text_length` had zero readers anywhere
    in the codebase as of iter 142 — never reached the inference path.
    Pin their absence so a copy-paste from old branches cannot revive
    them silently."""
    fields = set(RecognitionConfig.model_fields.keys())
    assert "max_input_width" not in fields
    assert "max_text_length" not in fields


def test_iter142_detection_arch_removed():
    """DBNet is hardcoded in pipeline.py; `DetectionConfig.arch` was a
    Literal field with zero readers. iter 142 removed it. If a future
    refactor wants pluggable detection backbones, do it via a factory
    function — a Literal in config that nothing reads is a lie."""
    from spinaiocr.config import DetectionConfig
    assert "arch" not in DetectionConfig.model_fields


def test_iter142_layoutconfig_dataconfig_removed():
    """Both `LayoutConfig` and `DataConfig` were scaffolding classes
    with no functional read in src/. Removed. Re-introduction must
    come with at least one consumer."""
    import spinaiocr.config as cfg_mod
    assert not hasattr(cfg_mod, "LayoutConfig")
    assert not hasattr(cfg_mod, "DataConfig")
    assert not hasattr(cfg_mod, "DEFAULT_DATA")
    assert not hasattr(cfg_mod, "DEFAULT_PIPELINE")


def test_iter142_pipeline_config_no_layout_field():
    """`PipelineConfig.layout` was the only non-trivial reference to
    `LayoutConfig` and read nothing. Pin that removing it does not
    silently leak via the schema."""
    from spinaiocr.config import PipelineConfig
    assert "layout" not in PipelineConfig.model_fields


# iter 145: easyocr_only routing mode — strict-winner opt-in.
# Bypasses v031 entirely; whole image goes to EasyOCR readtext.
# iter 143 real-photo bench (n=171): CER 22.87% vs accurate 26.11%
# vs v031-None 47.80%.

def test_easyocr_only_mode_accepted_by_config():
    """Pin the new Literal value is accepted by RecognitionConfig
    (and Pydantic rejects typos)."""
    cfg = RecognitionConfig(routing_mode="easyocr_only")
    assert cfg.routing_mode == "easyocr_only"
    with pytest.raises(Exception):
        RecognitionConfig(routing_mode="easyocronly")  # typo


def test_easyocr_only_returns_whole_image_text(monkeypatch):
    """easyocr_only mode must bypass v031 and return EasyOCR's own MULTI-REGION
    readtext output (polygon/text/conf per line), regardless of incoming v031
    polygons/texts/confidences. (2026-06: easyocr_only upgraded whole-image ->
    multi-region via _easyocr_readtext_multi; test updated to match committed
    behavior — was asserting the old whole-image _easyocr_fallback path.)"""
    pipe = _bare_pipeline("easyocr_only")
    eo1 = np.array([[5, 5], [60, 5], [60, 30], [5, 30]], dtype=np.float32)
    eo2 = np.array([[70, 5], [120, 5], [120, 30], [70, 30]], dtype=np.float32)
    monkeypatch.setattr(OCRPipeline, "_easyocr_readtext_multi",
                        lambda self, img: [(eo1, "EO_LINE1", 0.91), (eo2, "EO_LINE2", 0.88)],
                        raising=False)
    # Per-line replace must NEVER fire in this mode — patch a sentinel
    # that would change the output if it ran.
    monkeypatch.setattr(OCRPipeline, "_easyocr_recognize_line_crop",
                        lambda *a, **kw: "PER_LINE_SHOULD_NOT_FIRE", raising=False)

    img = np.zeros((120, 400, 3), dtype=np.uint8)
    polys = [
        np.array([[10, 10], [80, 10], [80, 40], [10, 40]], dtype=np.float32),
        np.array([[100, 10], [180, 10], [180, 40], [100, 40]], dtype=np.float32),
    ]
    texts = ["v031_a", "v031_b"]
    conf = np.array([0.99, 0.99], dtype=np.float32)  # high conf — must NOT preserve
    mcc = np.array([0.99, 0.99], dtype=np.float32)
    p2, t2, c2, m2 = pipe._apply_hybrid_routing(img, polys, texts, conf, mcc)
    # v031 inputs fully replaced by EasyOCR's multi-region output
    assert t2 == ["EO_LINE1", "EO_LINE2"]
    assert "PER_LINE_SHOULD_NOT_FIRE" not in t2
    assert len(p2) == 2
    assert len(c2) == 2 and len(m2) == 2


def test_vlm_mode_accepted_by_config():
    """`vlm` accuracy-tier mode must be a valid routing_mode (config Literal)."""
    cfg = RecognitionConfig(routing_mode="vlm")
    assert cfg.routing_mode == "vlm"


def test_vlm_preset_listed_in_routing_presets():
    pipe = _bare_pipeline("vlm")
    assert "vlm" in pipe._ROUTING_PRESETS
    assert pipe._resolve_routing_params() is not None


def test_vlm_uses_vl_recognition_per_crop(monkeypatch):
    """vlm mode: EasyOCR detects regions, PaddleOCR-VL recognizes each crop.
    Detected boxes are preserved; text comes from the VL service."""
    pipe = _bare_pipeline("vlm")
    eo1 = np.array([[5, 5], [60, 5], [60, 30], [5, 30]], dtype=np.float32)
    eo2 = np.array([[70, 5], [120, 5], [120, 30], [70, 30]], dtype=np.float32)
    monkeypatch.setattr(OCRPipeline, "_easyocr_readtext_multi",
                        lambda self, img: [(eo1, "EO_1", 0.5), (eo2, "EO_2", 0.5)],
                        raising=False)
    monkeypatch.setattr("spinaiocr.inference.vl_client.vl_healthy",
                        lambda timeout=3.0: True, raising=False)
    monkeypatch.setattr("spinaiocr.inference.vl_client.vl_ocr",
                        lambda crop, fallback="", **kw: "VL_TEXT", raising=False)
    img = np.zeros((120, 400, 3), dtype=np.uint8)
    polys = [np.array([[10, 10], [80, 10], [80, 40], [10, 40]], dtype=np.float32)]
    p2, t2, c2, m2 = pipe._apply_hybrid_routing(
        img, polys, ["v031"], np.array([0.99], np.float32), np.array([0.99], np.float32))
    assert t2 == ["VL_TEXT", "VL_TEXT"], "vlm must use VL recognition, not v031/EasyOCR text"
    assert len(p2) == 2, "EasyOCR-detected boxes preserved"


def test_vlm_degrades_to_easyocr_text_when_service_down(monkeypatch):
    """Invariant: when the VL service is unreachable, vlm mode returns the
    EasyOCR text per crop — never crashes, never worse than easyocr_only."""
    pipe = _bare_pipeline("vlm")
    eo1 = np.array([[5, 5], [60, 5], [60, 30], [5, 30]], dtype=np.float32)
    monkeypatch.setattr(OCRPipeline, "_easyocr_readtext_multi",
                        lambda self, img: [(eo1, "EO_FALLBACK", 0.5)], raising=False)
    monkeypatch.setattr("spinaiocr.inference.vl_client.vl_healthy",
                        lambda timeout=3.0: False, raising=False)
    img = np.zeros((120, 400, 3), dtype=np.uint8)
    polys = [np.array([[10, 10], [80, 10], [80, 40], [10, 40]], dtype=np.float32)]
    p2, t2, c2, m2 = pipe._apply_hybrid_routing(
        img, polys, ["v031"], np.array([0.99], np.float32), np.array([0.99], np.float32))
    assert t2 == ["EO_FALLBACK"], "VL down → degrade to EasyOCR text (graceful)"


def test_easyocr_only_falls_back_to_v031_when_easyocr_unavailable():
    """If EasyOCR import/init fails (poison sentinel), easyocr_only mode
    must NOT crash — return v031 inputs unchanged. Prevents the routing
    feature from making the pipeline strictly worse than v031."""
    pipe = _bare_pipeline("easyocr_only")
    pipe._easyocr_reader = False  # poison sentinel
    img = np.zeros((48, 200, 3), dtype=np.uint8)
    polys = [np.array([[10, 10], [100, 10], [100, 40], [10, 40]], dtype=np.float32)]
    texts = ["v031_text"]
    conf = np.array([0.65], dtype=np.float32)
    mcc = np.array([0.40], dtype=np.float32)
    p2, t2, c2, m2 = pipe._apply_hybrid_routing(img, polys, texts, conf, mcc)
    assert t2 == ["v031_text"], "no EasyOCR → v031 must pass through"
    assert np.array_equal(c2, conf)


def test_easyocr_only_preset_listed_in_routing_presets():
    """Pin that the mode is enrolled in `_ROUTING_PRESETS` so
    `_resolve_routing_params` returns a non-None tuple (not None,
    which would silently no-op)."""
    pipe = _bare_pipeline("easyocr_only")
    params = pipe._resolve_routing_params()
    assert params is not None
    assert params[0] == "easyocr_only"


# iter 164: image-level domain-aware routing tests.
# Hooks the subtitle-style classifier into balanced/accurate routing modes
# behind the routing_domain_aware opt-in flag. Default is OFF — adding the
# field must not change existing behavior. Subtitle-positive image must
# bypass v031 and route to EasyOCR readtext like easyocr_only does.

def test_routing_domain_aware_default_off():
    """Adding the field must NOT change consumer_v1 default behavior."""
    cfg = RecognitionConfig()
    assert cfg.routing_domain_aware is False


def test_routing_domain_aware_off_skips_classifier(monkeypatch):
    """When the flag is False (default), `_is_subtitle_style` MUST NOT be
    called — the classifier path is gated behind the opt-in."""
    pipe = _bare_pipeline("balanced")
    pipe.config.recognition.routing_domain_aware = False

    called = {"n": 0}
    def _spy(self, img):
        called["n"] += 1
        return True  # if it WERE called, it would (wrongly) fire
    monkeypatch.setattr(OCRPipeline, "_is_subtitle_style", _spy, raising=False)
    monkeypatch.setattr(OCRPipeline, "_easyocr_fallback",
                        lambda self, img: "DOMAIN_FALLBACK", raising=False)

    img = np.zeros((120, 400, 3), dtype=np.uint8)
    polys = [np.array([[10, 10], [80, 10], [80, 40], [10, 40]], dtype=np.float32)]
    texts = ["v031_text"]
    conf = np.array([0.99], dtype=np.float32)
    mcc = np.array([0.99], dtype=np.float32)
    _, t2, _, _ = pipe._apply_hybrid_routing(img, polys, texts, conf, mcc)
    assert called["n"] == 0
    assert t2 == ["v031_text"]


def test_routing_domain_aware_subtitle_positive_routes_to_easyocr(monkeypatch):
    """When flag is True and classifier returns True on a balanced/accurate
    call, output collapses to whole-image EasyOCR readtext (same shape as
    easyocr_only mode), regardless of v031 line confidences."""
    pipe = _bare_pipeline("balanced")
    pipe.config.recognition.routing_domain_aware = True
    monkeypatch.setattr(OCRPipeline, "_is_subtitle_style",
                        lambda self, img: True, raising=False)
    monkeypatch.setattr(OCRPipeline, "_easyocr_fallback",
                        lambda self, img: "DOMAIN_FALLBACK", raising=False)

    img = np.zeros((120, 400, 3), dtype=np.uint8)
    polys = [np.array([[10, 10], [80, 10], [80, 40], [10, 40]], dtype=np.float32)]
    texts = ["v031_high_conf"]
    conf = np.array([0.99], dtype=np.float32)  # high conf — must NOT preserve
    mcc = np.array([0.99], dtype=np.float32)
    p2, t2, c2, _ = pipe._apply_hybrid_routing(img, polys, texts, conf, mcc)
    assert t2 == ["DOMAIN_FALLBACK"]
    assert len(p2) == 1
    assert p2[0][2].tolist() == [400.0, 120.0]
    assert c2[0] == 1.0


def test_routing_domain_aware_subtitle_negative_falls_through(monkeypatch):
    """If classifier returns False, the rest of routing must run normally
    (per-line replace / cov fallback). Verified by spying on per-line."""
    pipe = _bare_pipeline("balanced")
    pipe.config.recognition.routing_domain_aware = True
    monkeypatch.setattr(OCRPipeline, "_is_subtitle_style",
                        lambda self, img: False, raising=False)
    monkeypatch.setattr(OCRPipeline, "_easyocr_recognize_line_crop",
                        lambda *a, **kw: "PER_LINE", raising=False)
    monkeypatch.setattr(OCRPipeline, "_easyocr_fallback",
                        lambda self, img: "WHOLE_FALLBACK", raising=False)

    img = np.zeros((120, 400, 3), dtype=np.uint8)
    polys = [np.array([[10, 10], [80, 10], [80, 40], [10, 40]], dtype=np.float32)]
    texts = ["v031_text"]
    # min_conf 0.50 < cov_thresh 0.65 (balanced) → whole-image fallback fires
    conf = np.array([0.50], dtype=np.float32)
    mcc = np.array([0.50], dtype=np.float32)
    _, t2, _, _ = pipe._apply_hybrid_routing(img, polys, texts, conf, mcc)
    # Whole-image fallback fired (not domain-aware bypass) — text matches
    # _easyocr_fallback patch.
    assert t2 == ["WHOLE_FALLBACK"]


def test_routing_domain_aware_no_op_in_fast_mode(monkeypatch):
    """fast mode lacks whole-image escalation — adding domain-aware
    behaviour to it would silently break the fast preset's contract.
    Pin that the dispatch is gated to balanced/accurate only."""
    pipe = _bare_pipeline("fast")
    pipe.config.recognition.routing_domain_aware = True
    called = {"n": 0}
    def _spy(self, img):
        called["n"] += 1
        return True
    monkeypatch.setattr(OCRPipeline, "_is_subtitle_style", _spy, raising=False)
    monkeypatch.setattr(OCRPipeline, "_easyocr_fallback",
                        lambda self, img: "DOMAIN_FALLBACK", raising=False)

    img = np.zeros((120, 400, 3), dtype=np.uint8)
    polys = [np.array([[10, 10], [80, 10], [80, 40], [10, 40]], dtype=np.float32)]
    texts = ["v031_text"]
    conf = np.array([0.99], dtype=np.float32)
    mcc = np.array([0.99], dtype=np.float32)
    _, t2, _, _ = pipe._apply_hybrid_routing(img, polys, texts, conf, mcc)
    assert called["n"] == 0
    assert t2 == ["v031_text"]


def test_is_subtitle_style_rejects_grayscale_or_tiny():
    """Defensive guards: tiny / non-RGB images must classify as False so
    the flag never accidentally triggers on inputs the heuristic was not
    calibrated for."""
    pipe = _bare_pipeline("balanced")
    # Tiny RGB
    assert pipe._is_subtitle_style(np.zeros((8, 8, 3), dtype=np.uint8)) is False
    # Non-RGB (grayscale 2D)
    assert pipe._is_subtitle_style(np.zeros((48, 200), dtype=np.uint8)) is False
    # 4-channel (e.g. RGBA)
    assert pipe._is_subtitle_style(np.zeros((48, 200, 4), dtype=np.uint8)) is False
    # Plain black RGB (no text, ring density 0)
    assert pipe._is_subtitle_style(np.zeros((48, 200, 3), dtype=np.uint8)) is False
