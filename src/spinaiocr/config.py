"""Pydantic-based config for SPINAI OCR."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ModelTier = Literal["lite", "standard", "large", "medical", "consumer_v1"]
LangCode = Literal["ko", "en", "ja", "zh", "multi"]


class DetectionConfig(BaseModel):
    backbone: str = "resnet18"
    # _70 (iter 13): default 960 → 480. DBNet was trained at 480-ish
    # resolution (_38 note) and at 960 the upsampled image triggers
    # character-fragment detections on multi-line pages (iter 13 bench
    # found 9 micro-regions for a 3-line image — 8× over-segmentation).
    # 480 matches training → fewer spurious small boxes.
    # _148 (iter 93): re-confirmed at 480/640/800/1024 sweep on n=50
    # native synth + simulated big-photo (2400x2400 JPEG). 480 wins
    # both monotonically. Above 480, precision dives 0.95→0.54 from
    # over-fragmentation regardless of resize direction. Effect is
    # input_size-vs-training-resolution, not upsample-specific. Locked.
    backend: Literal["dbnet", "craft"] = "dbnet"
    input_size: int = 480
    # _122 (iter 68): det_ko F1 sweep found 0.3/0.7 thresh combo improves
    # detection quality (F1 0.9468 → 0.9763, +3.1%) without sacrificing
    # recognition. Lower thresh catches thin h=21-23 boxes (32% of iter 67
    # misses), higher box_thresh filters spurious.
    #
    # iter 113 unclip + recognize_extra_ratio decoupling: detect at unclip=2.0
    # (tight polys for F1 + return), then dilate each tight poly by
    # `recognize_extra_ratio * area/perim` BEFORE _crop_polygon for rec input.
    # Bench (iter 112, 3-fixture cross-check on synth det_ko): F1 +2.49pp avg
    # (largest single-iter F1 lift since iter 67 baseline), composite +0.0173
    # all positive. mc trade-off: +0.0070 abs avg (+22% rel), still under 0.05
    # absolute. Lifts the iter 109-diagnosed F1 ceiling 0.9525 → 0.9783.
    thresh: float = 0.3
    box_thresh: float = 0.7
    unclip_ratio: float = 2.0
    # iter 113: post-detection dilation factor for rec-input cropping. 0.0 =
    # no decouple (return polys = rec-crop polys, legacy behavior). 0.4 =
    # iter 112 measured Pareto-positive (e ∈ [0.3, 0.5] F1-tied plateau,
    # e=0.4 mc minimum). e≥0.6 harms F1 via crop-merge.
    recognize_extra_ratio: float = 0.4
    # iter 80: width-aware unclip. After uniform pyclipper offset, optionally
    # expand horizontally by `wide_horizontal_extra` fraction on each side
    # when box aspect ratio ≥ wide_ar_thresh. Default 0.0 = no change (back-compat).
    # iter 79 audit: 97% of det misses are wide ar≥3 spans where uniform unclip
    # under-extends horizontally (IoU<0.5 dropoff at GT polygon edges).
    wide_ar_thresh: float = 3.0
    wide_horizontal_extra: float = 0.0


class RecognitionConfig(BaseModel):
    arch: Literal["crnn", "svtr", "svtr_lite", "svtr_wide", "parseq", "vitstr"] = "svtr_lite"
    input_height: int = 48
    vocab_name: str = "ko_en_v1"
    # _74 (iter 17): post-hoc calibration exponent applied to per-line
    # confidence. Raw CTC conf = geometric mean of top-1 softmax prob
    # over steps; well known to be overconfident on short sequences.
    # Live bench (n=96, v2d iter7 fonts): raw mean conf 0.948 vs char-acc
    # 0.837 (ECE 0.110). Fitting conf**p=acc yields p≈3.3; p=3.0 drops
    # ECE to 0.021 (−81%) and MCE to 0.045. 1.0 = no rescaling.
    conf_power: float = 1.0
    # iter 53: subtract this constant from logits[blank_id] before greedy
    # argmax, suppressing premature CTC blank emission. Diagnostic on
    # cleaned held-out (n=262) showed mean blank_top1_rate=0.67; sweep
    # found optimum at 0.4 (mean CER 0.3498→0.3436, -1.8% on 92 high-blank
    # rows -1.0pp absolute). 0.0 = no change (backward compatible).
    blank_penalty: float = 0.0
    # iter 141: `easyocr_fallback_threshold` REMOVED. Superseded by
    # `routing_mode` (iter 131/132). consumer_v1 default behavior is
    # now `routing_mode="balanced"` set in OCRPipeline.__init__ via
    # the _TIER_DEFAULT_ROUTING_MODE map. iter 57/59 single-threshold
    # whole-line fallback was Pareto-dominated by the iter 130 per-line
    # routing path on n=500 OOD det_ko_v2.
    # iter 131: productionize three-mode hybrid routing (iter 128/129/130
    # research). None = legacy (no routing). When set, multi-poly path runs
    # v031 first, then per-line replaces low-conf lines via EasyOCR
    # recognize-direct on bbox crop, and (modes balanced/accurate) escalates
    # to whole-image EasyOCR readtext when v031 coverage signal is below
    # routing_cov_thresh. Mode presets:
    #   fast      → line_thresh=0.70, cov off                  (iter129)
    #   balanced  → line_thresh=0.70, cov=min_conf<0.65→whole  (iter130)
    #   accurate  → line_thresh=0.90, cov=min_conf<0.80→whole  (iter130)
    # User overrides via routing_line_thresh / routing_cov_thresh /
    # routing_cov_signal still take priority. EasyOCR is an optional dep —
    # if import fails, routing degrades silently to v031-only output.
    routing_mode: Literal["fast", "balanced", "accurate", "easyocr_only", "vlm"] | None = None
    routing_line_thresh: float | None = None
    routing_cov_thresh: float | None = None
    routing_cov_signal: Literal["min_conf", "mean_conf"] = "min_conf"
    # iter 164: image-level domain-aware routing. When True and routing_mode in
    # {balanced, accurate}, classify the input image as "broadcast subtitle
    # style" via saturation-p90 + Otsu-outline-ring density. Subtitle-classified
    # images skip v031 entirely and go directly to EasyOCR readtext, since
    # iter 162 established that synthetic aug saturates at routing-calibration
    # only and iter 163 closed the real-subtitle-data lever. Opt-in (default
    # False) so existing prod behavior is unchanged.
    routing_domain_aware: bool = False
    inference_max_width: int = 640
    # harness flags (iter 166)
    # low_conf_upscale_retry: re-run recognition on 1.5x upscaled crop when
    #   first-pass confidence < 0.5. Free quality gain, no model change needed.
    low_conf_upscale_retry: bool = True
    # use_jamo_correction: apply rule-based Korean jamo post-correction after
    #   all recognition + LM decoding. Fixes split-jamo and spurious spaces.
    use_jamo_correction: bool = True


DecodeMode = Literal["beam_lm", "greedy"]


class PipelineConfig(BaseModel):
    lang: LangCode = "ko"
    tier: ModelTier = "lite"
    device: Literal["cpu", "cuda", "auto"] = "auto"
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    recognition: RecognitionConfig = Field(default_factory=RecognitionConfig)
    use_llm_postprocess: bool = False
    # "beam_lm" = best quality, ~130 ms decode for 8 crops.
    # "greedy"  = ~10× faster, small quality drop on long lines. Tradeoff
    # exposed for latency-sensitive callers (e.g. /ocr?decode=greedy).
    decode_mode: DecodeMode = "beam_lm"
