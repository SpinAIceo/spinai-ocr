"""End-to-end OCR pipeline.

Stages:
    1. Preprocess (resize, optional CLAHE).
    2. Detect text polygons (DBNet).
    3. Refine polygons (edge snap; optional DexiNed backend).
    4. Crop each polygon and recognize text (CRNN / SVTR).
    5. Optional LLM post-correction (Claude Haiku by default).
    6. Optional layout analysis → structured output.

Models are loaded lazily from ``checkpoints/<tier>/<lang>/{det,rec}.pth``.
If no checkpoint is present, the pipeline runs in degraded mode (returns
empty ``lines``) so serving and CI can still boot.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from spinai_ocr.config import PipelineConfig
from spinai_ocr.data.augment import clahe
from spinai_ocr.inference.db_postprocess import DBPostProcessor
from spinai_ocr.inference.decoders import BeamConfig, CharBigramLM, decode_batch
from spinai_ocr.inference.edge_refine import EdgeRefiner, SnapConfig
from spinai_ocr.log import get_logger, log_span
from spinai_ocr.models.detection import DBNet
from spinai_ocr.models.recognition import build_recognition
from spinai_ocr.vocab.base import Vocab, load_vocab

log = get_logger("spinai_ocr.inference")


def _strip_leading_dupe(t: str) -> str:
    """iter 123: drop a single repeated prefix group when rec emits a CTC-style
    leading duplicate. Targets the 70%-of-22% DET_BOUNDARY long-bucket failure
    surface (audit_iter123_boundary_mechanism.json).

    Rule: for k ∈ {3, 2, 1}, if s[:k] == s[k:2k] AND s[:k] != s[2k:3k] (not a
    legitimate triple-repeat), drop the leading k chars. Length floor 6 keeps
    short tokens like 'BBQ' / 'ABBA' / 'AAA' untouched. Off by default; toggle
    with SPINAI_LEADING_DUPE_STRIP=1 (env). When off, returns input unchanged.
    """
    import os as _os
    if _os.environ.get("SPINAI_LEADING_DUPE_STRIP") != "1":
        return t
    s = t.strip()
    if len(s) < 6:
        return t
    for k in (3, 2, 1):
        if 2 * k > len(s):
            continue
        if any(c.isspace() for c in s[:k]):
            continue  # word-boundary prefix (e.g., '정말 정말') — legitimate repeat
        if s[:k] != s[k:2 * k]:
            continue
        if 3 * k <= len(s) and s[:k] == s[2 * k:3 * k]:
            continue  # triple+ repeat at this k — bail rather than corrupt
        result = s[k:]
        if len(result) >= 2 and result[0] == result[1]:
            continue  # post-drop still char-repeat → underlying long repeat ('하하하하하'), bail
        lead_ws = t[:len(t) - len(t.lstrip())]
        return lead_ws + result
    return t


# LM fusion weight — **distribution-sensitive** (finding from _34, _36, _38).
#
# single_line (pre-cropped clean lines, e.g. detcrops_v1):
#     Recognizer posteriors are peaky; the char-LM adds noise. Sweep on
#     detcrops_v1 n=264: α=0.3 → CER 0.0591, α=0.05 → 0.0366. After _38
#     korean-pair corpus expansion (4844→7826 lines), best α moves slightly:
#     α=0.05 → 0.0353, **α=0.1 → 0.0350** (new best).
#
# multi-polygon (full-page detect-then-recognize, e.g. det_ko pages):
#     Noisy detector gives spurious/warped crops. Recognizer hallucinates
#     on low-quality crops; the LM filters/corrects. _36 E2E bench at n=30:
#     α=0.3 → bag_char_F1 0.4142 → still best. korean-pair corpus actually
#     hurts E2E (-2.5% F1), so multi_poly LM keeps the original corpus.
#
# Picking a single global α forces a trade-off: one path wins, the other
# regresses. Branch-specific defaults (and now branch-specific LMs) let both
# win.
#
# iter 107 RE-SWEEP (multi-poly only): n=200 synth det_ko, α ∈ {0.3, 0.5,
#     0.6, 0.7}. Crop quality improved meaningfully since iter 36 baseline
#     (iter 100 _crop_polygon warp+resize fusion + iter 101 union-bbox
#     refine + iter 102 half-res Sobel + iter 104 T-slice). Cleaner crops
#     mean LM signal cuts through more — optimum **shifted from 0.3 → 0.6**.
#     F1 byte-tie 0.9525 across 0.3/0.5/0.6 (n_pred 1357 identical),
#     matched_cer 0.0450 (α=0.3) → 0.0340 (α=0.6) = **-24.4% rel WIN**.
#     α=0.7 starts to over-correct (matched_cer rebounds 0.0374, F1 jitter).
#     Reproduced 4× back-to-back: F1 byte-identical, matched_cer Δ exactly
#     -0.0092 every run.
#
# iter 108 RE-SWEEP (single_line only): n=100 detcrops_ko sample100 +
#     n=200 held-out, α ∈ {0.05 … 1.5}. Same precedent rationale as iter 107:
#     iter 38 set α=0.1 with then-current decode; cumulative iter 100/101/102/
#     104 quality lifts (and the now-extended LM with korean-pair+detcrops
#     bigrams) widened the headroom for stronger LM weighting.
#     Plateau identified at α ∈ [0.45, 0.6]: CER 0.0103 byte-tied across the
#     plateau on sample100. α=0.65 starts to slip (0.0121); α=0.7 over-corrects
#     (0.0181, +75% vs plateau); α=1.0 catastrophic (0.18). Held-out 200
#     selects α=0.5 cleanly (others within plateau slip there). **0.1 → 0.5
#     = -54.0% rel CER on sample100 / -50.8% rel on holdout200**. Picked 0.5
#     (middle of plateau, both samples agree).
LM_ALPHA_SINGLE_LINE = 0.5
LM_ALPHA_MULTI_POLY = 0.6

# iter 85 set bw=4 from {2,4,6,8} sweep on then-current decode (LM α=0.1/0.3,
# bp=0.4, v021 ckpt). iter 95 LOCKED bw=4 optimum on same-era stack.
#
# iter 111 RE-SWEEP (v022 + α=0.5/0.6 + bp=1.9 regime, 4-fixture cross-check):
#   single_line: bw=2 catastrophic on holdout200 (CER 0.0133, +18.8% rel),
#     bw=4 baseline (0.0112), bw≥5 plateau at 0.0103 (-8.0% rel WIN). Sample100
#     byte-tied 0.0103 across all bw≥2 (plateau-saturated, fixture too clean).
#     Holdout200 disambiguates: bw=5 closes the sample100/holdout gap.
#   multi-poly: bw=4 retained as iter 95 lock — bw=5 gives F1 +0.0008 / mc
#     +0.0005 (within run-to-run jitter) at +41% p50 latency. Not Pareto-positive.
#   Beam search is fully deterministic — confirmed across 3 reruns (CER bytetie).
# Split config (decode-fixture-aware, same pattern as LM_ALPHA_*).
BEAM_WIDTH_SINGLE_LINE = 5
BEAM_WIDTH_MULTI_POLY = 4
BEAM_TOPK = 10


ImageLike = Union[str, Path, np.ndarray, Image.Image]


@dataclass
class OCRLine:
    text: str
    bbox: list[tuple[float, float]]
    confidence: float
    lang: str = "ko"
    # _53: medical safety — raw <unk> counts per line so API / UI can
    # reject or queue rather than silently replacing OOV chars.
    unk_count: int = 0
    unk_positions: list[int] = field(default_factory=list)
    # _80 (iter 24): min per-step top-1 softmax prob across the line.
    # `confidence` is the geometric mean (after per-tier calibration); this
    # is the single weakest step, which is the right signal for "highlight
    # the suspicious character in the UI". Same conf_power exponent is
    # applied so values are comparable in scale.
    min_char_conf: float = 1.0


@dataclass
class OCRResult:
    lines: list[OCRLine] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0
    timing_ms: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n".join(l.text for l in self.lines)


def _to_numpy(x: ImageLike) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, Image.Image):
        x = ImageOps.exif_transpose(x)
        return np.asarray(x.convert("RGB"))
    img = Image.open(x)
    img = ImageOps.exif_transpose(img)
    return np.asarray(img.convert("RGB"))


def _resolve_device(choice: str) -> str:
    if choice == "auto":
        # `is_available()` can return True even when device_count() is 0
        # (e.g. CUDA_VISIBLE_DEVICES="" with already-initialized torch).
        # Both must hold or torch.load will fail to map storage off cuda:0.
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return "cuda"
        return "cpu"
    return choice


def _resize_keep_aspect(img: np.ndarray, target: int) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    scale = target / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    # pad to multiple of 32 for detection backbone
    nh = ((nh + 31) // 32) * 32
    nw = ((nw + 31) // 32) * 32
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return resized, scale


# Module-level ImageNet stats — broadcast-shaped for (H, W, 3) input.
# Precompute the (mean, inv_std) pair in float32 so the in-place detector
# normalisation stays in float32 (previously two np.array calls per /ocr).
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
_IMAGENET_INV_STD = (
    np.float32(1.0) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
).reshape(1, 1, 3)
# iter 99: pre-fuse the /255 into INV_STD and the mean into the
# subtract operand. The transform `(x/255 - mean) * inv_std` is
# algebraically identical to `x * (inv_std/255) - mean*inv_std`.
# Folding /255 into A drops one full-tensor pass (was 4 arithmetic
# passes: astype, *=1/255, -=MEAN, *=INV_STD; now 3: astype, *=A,
# -=B). Numerical max abs diff vs baseline 4.77e-7 (single ULP),
# well below the network's float32 forward-pass noise floor. Bench
# (n=2000): det 480x480x3 mean 2.41 → 2.35ms (-2.5%); rec 48x320x3
# mean 0.116 → 0.112ms (-3.4%).
_FUSED_A = (_IMAGENET_INV_STD * np.float32(1.0 / 255.0)).astype(np.float32)
_FUSED_B = (_IMAGENET_MEAN * _IMAGENET_INV_STD).astype(np.float32)


def _to_tensor(img: np.ndarray, device: str) -> torch.Tensor:
    t = img.astype(np.float32)
    t *= _FUSED_A
    t -= _FUSED_B
    t = torch.from_numpy(np.ascontiguousarray(t.transpose(2, 0, 1))).unsqueeze(0).to(device)
    return t


def _crop_polygon(img: np.ndarray, polygon: np.ndarray, target_h: int = 48) -> np.ndarray:
    """Perspective-warp a quadrilateral to a horizontal strip.

    iter 100: fused warp+resize. Was warpPerspective→(w,h) then
    cv2.resize→(new_w, target_h) — two interpolation passes that
    materialized a large intermediate crop on big-photo. Now warps
    directly to (new_w, target_h). Bench: native (480x480, 11
    polys) 0.75ms tie; big (2400x2400, 11 polys) 4.52→0.82ms (-82%).
    Recognizer text output unchanged across n=50 (max pixel diff 12,
    mean 0.71 — bilinear rounding noise well below network input
    sensitivity). Saves ~3.7ms per big-photo /ocr.
    """
    poly = polygon.astype(np.float32)
    widths = [np.linalg.norm(poly[0] - poly[1]), np.linalg.norm(poly[2] - poly[3])]
    heights = [np.linalg.norm(poly[0] - poly[3]), np.linalg.norm(poly[1] - poly[2])]
    w = int(max(widths))
    h = int(max(heights))
    if w <= 0 or h <= 0:
        return np.zeros((target_h, target_h, 3), dtype=np.uint8)
    scale = target_h / max(h, 1)
    new_w = max(int(w * scale), 8)
    dst = np.array([[0, 0], [new_w, 0], [new_w, target_h], [0, target_h]],
                    dtype=np.float32)
    M = cv2.getPerspectiveTransform(poly, dst)
    return cv2.warpPerspective(img, M, (new_w, target_h))


def _merge_horizontal_boxes(
    polygons: list[np.ndarray],
    *,
    y_overlap_thresh: float = 0.5,
    x_gap_ratio: float = 1.0,
    height_ratio: float = 2.0,
) -> list[np.ndarray]:
    """Merge neighboring polygons that are on the same text line.

    Two boxes are merged into one (convex hull) when all hold:
      - Y-overlap >= y_overlap_thresh * min(h_a, h_b)  [same line]
      - X-gap     <  x_gap_ratio * min(h_a, h_b)       [close enough]
      - h_a / h_b in [1/height_ratio, height_ratio]   [not mixed sizes]

    Iterates until no more merges happen. Polygon is kept as an Nx2 ndarray
    (original quad form). The hull is re-rectified to a 4-point min-area
    rect in the caller's coordinate system.

    _75 (iter 19): added to attack E2E p95 tail — over-segmented regions
    were the dominant failure mode after unclip=3.0.
    """
    if len(polygons) <= 1:
        return polygons

    def _aabb(p: np.ndarray) -> tuple[float, float, float, float]:
        return (float(p[:, 0].min()), float(p[:, 1].min()),
                float(p[:, 0].max()), float(p[:, 1].max()))

    boxes = [_aabb(p) for p in polygons]
    alive = [True] * len(polygons)
    polys = [p.copy() for p in polygons]

    merged_any = True
    while merged_any:
        merged_any = False
        for i in range(len(polys)):
            if not alive[i]:
                continue
            for j in range(i + 1, len(polys)):
                if not alive[j]:
                    continue
                (xi0, yi0, xi1, yi1) = boxes[i]
                (xj0, yj0, xj1, yj1) = boxes[j]
                hi, hj = yi1 - yi0, yj1 - yj0
                if hi <= 0 or hj <= 0:
                    continue
                min_h = min(hi, hj)
                # Y-overlap
                y_overlap = max(0.0, min(yi1, yj1) - max(yi0, yj0))
                if y_overlap < y_overlap_thresh * min_h:
                    continue
                # X-gap (negative means overlapping in X — also merge)
                if xi1 <= xj0:
                    x_gap = xj0 - xi1
                elif xj1 <= xi0:
                    x_gap = xi0 - xj1
                else:
                    x_gap = -1.0  # overlapping
                if x_gap >= x_gap_ratio * min_h:
                    continue
                # Size ratio — don't merge a tall box with a tiny one
                ratio = hi / hj if hi >= hj else hj / hi
                if ratio > height_ratio:
                    continue
                # Merge: min-area rect over all points. Keep 4-point quad
                # so downstream `_crop_polygon` (cv2.getPerspectiveTransform)
                # still gets a valid quadrilateral.
                merged_pts = np.concatenate([polys[i], polys[j]], axis=0).astype(np.float32)
                rect = cv2.minAreaRect(merged_pts)
                quad = cv2.boxPoints(rect)  # 4x2, float32
                # Canonical order: top-left, top-right, bottom-right, bottom-left
                order = np.argsort(quad[:, 0])
                left = quad[order[:2]]
                right = quad[order[2:]]
                left = left[np.argsort(left[:, 1])]
                right = right[np.argsort(right[:, 1])]
                polys[i] = np.array([left[0], right[0], right[1], left[1]], dtype=np.float32)
                boxes[i] = _aabb(polys[i])
                alive[j] = False
                merged_any = True
        # Filter dead boxes after each full pass so subsequent iters are smaller
        polys = [p for p, a in zip(polys, alive) if a]
        boxes = [b for b, a in zip(boxes, alive) if a]
        alive = [True] * len(polys)

    return polys


def _split_tall_boxes(
    img: np.ndarray,
    polygons: list[np.ndarray],
    *,
    aspect_thresh: float = 1.5,
    min_line_h: int = 15,
    img_area: float = 0.0,
    min_area_frac: float = 0.30,
) -> list[np.ndarray]:
    """Split tall detected boxes into horizontal text lines using projection.

    Only activates when a polygon covers a large fraction of the image
    (≥min_area_frac) AND is taller than wide (≥aspect_thresh). This
    prevents splitting normal tall/narrow text regions like vertical signs.
    """
    out: list[np.ndarray] = []
    for poly in polygons:
        x0, y0 = int(poly[:, 0].min()), int(poly[:, 1].min())
        x1, y1 = int(poly[:, 0].max()), int(poly[:, 1].max())
        bw, bh = x1 - x0, y1 - y0
        if bh <= 0 or bw <= 0 or bh < aspect_thresh * bw:
            out.append(poly)
            continue
        poly_area = float(bw * bh)
        if img_area > 0 and poly_area / img_area < min_area_frac:
            out.append(poly)
            continue
        crop = img[max(y0, 0):min(y1, img.shape[0]),
                   max(x0, 0):min(x1, img.shape[1])]
        if crop.size == 0:
            out.append(poly)
            continue
        gray = np.mean(crop, axis=2) if crop.ndim == 3 else crop.astype(float)
        proj = np.mean(gray, axis=1)
        thresh = np.percentile(proj, 85)
        is_gap = proj > thresh
        lines: list[tuple[int, int]] = []
        in_text = False
        start = 0
        for r in range(len(is_gap)):
            if not is_gap[r] and not in_text:
                in_text = True
                start = r
            elif is_gap[r] and in_text:
                if r - start >= min_line_h:
                    lines.append((start, r))
                in_text = False
        if in_text and len(is_gap) - start >= min_line_h:
            lines.append((start, len(is_gap)))
        if len(lines) <= 1:
            out.append(poly)
            continue
        for ly0, ly1 in lines:
            line_poly = np.array([
                [x0, y0 + ly0], [x1, y0 + ly0],
                [x1, y0 + ly1], [x0, y0 + ly1],
            ], dtype=np.float32)
            out.append(line_poly)
    return out


def _recognition_tensor(crops: list[np.ndarray], device: str) -> tuple[torch.Tensor, list[int]]:
    if not crops:
        return torch.zeros(0, 3, 1, 1, device=device), []
    max_w = max(c.shape[1] for c in crops)
    h = crops[0].shape[0]
    # Pre-allocate the batch buffer (white-filled) and blit crops in —
    # avoids per-crop np.concatenate + np.stack allocations, which were
    # 2–3x overhead for wide polygon lists.
    # Match training preprocessing: /255 only, NO ImageNet normalization.
    batch = np.full((len(crops), h, max_w, 3), 255, dtype=np.uint8)
    widths = [0] * len(crops)
    for i, c in enumerate(crops):
        w = c.shape[1]
        batch[i, :, :w, :] = c
        widths[i] = w
    batch_f = batch.astype(np.float32) / 255.0
    t = torch.from_numpy(batch_f.transpose(0, 3, 1, 2)).contiguous().to(device)
    return t, widths


def _ctc_greedy_decode(logits: torch.Tensor, vocab: Vocab) -> list[str]:
    ids = logits.argmax(dim=-1).cpu().numpy()
    return [vocab.decode(row, ctc_collapse=True) for row in ids]


class OCRPipeline:
    def __init__(
        self,
        lang: str = "ko",
        tier: str = "lite",
        device: str = "auto",
        config: PipelineConfig | None = None,
        checkpoints_root: str | Path = "checkpoints",
    ) -> None:
        self.config = config or PipelineConfig(lang=lang, tier=tier, device=device)
        # _74 (iter 17), _77 (iter 21): per-tier default conf calibration fit
        # from iter7 96-sample bench (ECE before/after):
        #   consumer_v1: p=3.0  (0.110 -> 0.021, -81%)  iter 17
        #   lite:        p=8.0  (0.456 -> 0.057, -88%)  iter 21
        #   medical:     p=6.0  (0.278 -> 0.037, -87%)  iter 21
        # Applied only when user left conf_power at the default 1.0.
        _TIER_CONF_POWER = {
            "consumer_v1": 3.0,
            "lite":        8.0,
            "medical":     6.0,
        }
        if self.config.recognition.conf_power == 1.0:
            self.config.recognition.conf_power = _TIER_CONF_POWER.get(
                self.config.tier, 1.0
            )
        # iter 53: per-tier blank_penalty fit from cleaned held-out (n=262).
        # consumer_v1 sweep optimum at 0.4 (-1.8% mean CER, biggest gain on
        # high-blank-rate rows where model emits blank prematurely). Other
        # tiers untested → 0.0 (safe default). Applied only when user left
        # blank_penalty at the default 0.0.
        # iter 110 RE-FIT (consumer_v1 only): post v022 ckpt swap (iter 89)
        #     + LM_ALPHA re-fits (iter 107: multi-poly 0.3→0.6, iter 108:
        #     single_line 0.1→0.5). Decode-side HP drift pattern (iter 109
        #     lock) suggested re-fit. Sweep on detcrops_ko sample100 + 200
        #     held-out + synth det_ko n=100 + n=100 held-out (4-fixture
        #     cross-check). bp=2.0 overfits sample100 (single_line CER 0.0095
        #     vs 0.0155 on held-out — fixture-specific cliff). bp=1.9 = clean
        #     Pareto improvement on all 4 fixtures: matched_cer 0.0353 → 0.0271
        #     (-23% rel main) / 0.0431 → 0.0393 (-9% rel held); F1 +0.0007
        #     main / +0.0057 held; single_line non-regressing on both samples.
        _TIER_BLANK_PENALTY = {
            "consumer_v1": 1.9,
        }
        if self.config.recognition.blank_penalty == 0.0:
            self.config.recognition.blank_penalty = _TIER_BLANK_PENALTY.get(
                self.config.tier, 0.0
            )
        # iter 141: per-tier default routing_mode (replaces iter 57/59
        # `_TIER_FALLBACK` whole-line fallback that was Pareto-dominated by
        # iter 130 per-line routing on synthetic OOD).
        # iter 144: consumer_v1 default escalated `balanced` → `accurate`
        # after iter 143 honest bench on real Korean photos (n=171,
        # `data/korean_pair_heldout_test/labels_singleline_clean.tsv`)
        # revealed `balanced` is essentially a no-op on real photos
        # (CER 47.80% = same as None; line/cov thresholds rarely fire
        # because v031 conf is mis-calibrated on this domain). `accurate`
        # ships -45.4% rel CER (47.80% → 26.11%) at p50 8→28 ms — within
        # interactive SLO. iter 130 synthetic ranking inverted on real
        # data; the synthetic surrogate did not transfer.
        _TIER_DEFAULT_ROUTING_MODE = {
            "consumer_v1": "accurate",
        }
        if self.config.recognition.routing_mode is None:
            self.config.recognition.routing_mode = (
                _TIER_DEFAULT_ROUTING_MODE.get(self.config.tier, None)
            )
        self._easyocr_reader = None  # lazy-init on first routing trigger
        self.device = _resolve_device(self.config.device)
        self.checkpoints_root = Path(checkpoints_root)
        self._detector: DBNet | None = None
        self._recognizer: torch.nn.Module | None = None
        # _84 (iter 27): optional secondary recognizer for v2d + v2f ensemble.
        # Loaded from `checkpoints/<tier>/<lang>/rec_v2d_pre_iter20.pth.bak`
        # when SPINAI_ENSEMBLE=1 is set. v2d (no-aug) + v2f (aug) have
        # complementary error modes → log-prob averaging typically trims
        # a bit of CER for free.
        self._recognizer_secondary: torch.nn.Module | None = None
        self._vocab: Vocab = load_vocab(self.config.recognition.vocab_name)
        self._refiner = EdgeRefiner(SnapConfig(backend="sobel"))
        self._post_det = DBPostProcessor(
            thresh=self.config.detection.thresh,
            box_thresh=self.config.detection.box_thresh,
            unclip_ratio=self.config.detection.unclip_ratio,
            wide_ar_thresh=self.config.detection.wide_ar_thresh,
            wide_horizontal_extra=self.config.detection.wide_horizontal_extra,
            recognize_extra_ratio=self.config.detection.recognize_extra_ratio,
        )
        self._llm_corrector = None  # lazy
        self._angle_classifier = None  # lazy; loaded only if ckpt present
        # Two LMs: _lm_single_line includes domain-shifted korean-pair (+detcrops),
        # _lm_multi_poly excludes it (+E2E pages). _38 split decision.
        self._lm_single_line: CharBigramLM | None = None  # lazy
        self._lm_multi_poly: CharBigramLM | None = None  # lazy

    # ----- model loading -------------------------------------------------

    def _load_checkpoint_if_present(self, name: str, model: torch.nn.Module) -> bool:
        ckpt_path = (
            self.checkpoints_root / self.config.tier / self.config.lang / f"{name}.pth"
        )
        if not ckpt_path.exists():
            # iter 157 guard: bench scripts that pass an explicit
            # `--checkpoints-root <staging>` should opt into strict mode so
            # a missing rec.pth raises rather than silently running an
            # untrained model — this is what produced the iter 156 false
            # catastrophic-NEG (v037 was reported CER=1.0 because the staged
            # path had `<root>/ko/rec.pth` not `<root>/consumer_v1/ko/rec.pth`).
            import os as _os
            if _os.environ.get("SPINAI_STRICT_CKPT", "").lower() in ("1", "true", "yes"):
                raise FileNotFoundError(
                    f"SPINAI_STRICT_CKPT: required ckpt missing at {ckpt_path}"
                )
            log.warning("ckpt.missing name=%s path=%s — running untrained (no lines will be returned)",
                        name, ckpt_path,
                        extra={"ckpt_name": name, "path": str(ckpt_path)})
            return False
        try:
            raw = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            state = raw
            if isinstance(state, dict):
                # If the recognizer checkpoint saved its own vocab_chars, rebuild
                # the Vocab from them so index↔char mapping matches training.
                if name == "rec" and "vocab_chars" in state:
                    self._vocab = Vocab(
                        name=state.get("vocab_name", self.config.recognition.vocab_name),
                        chars=state["vocab_chars"],
                    )
                    log.info("vocab.overridden_from_ckpt size=%d", self._vocab.size,
                             extra={"ckpt_name": name, "vocab_size": self._vocab.size})
                if "state_dict" in state:
                    state = state["state_dict"]
                state = {k.replace("student.", "", 1): v for k, v in state.items()}
            missing, unexpected = model.load_state_dict(state, strict=False)
            log.info("ckpt.loaded name=%s path=%s missing_keys=%d unexpected_keys=%d",
                     name, ckpt_path, len(missing), len(unexpected),
                     extra={"ckpt_name": name, "path": str(ckpt_path),
                            "missing_keys": len(missing), "unexpected_keys": len(unexpected)})
            return True
        except Exception as e:
            log.error("ckpt.load_failed name=%s path=%s: %s", name, ckpt_path, e,
                      exc_info=True, extra={"ckpt_name": name, "path": str(ckpt_path)})
            return False

    # Cached base-corpus read so the TWO _build_lm calls (single_line /
    # multi_poly) don't duplicate disk I/O + parsing. _51.
    _BASE_CACHE: list[str] | None = None
    _EXT_CACHE: list[str] | None = None

    @classmethod
    def _load_base_corpus(cls) -> list[str]:
        if cls._BASE_CACHE is not None:
            return cls._BASE_CACHE
        from spinai_ocr.data.synth_simple import default_korean_corpus
        lines: list[str] = list(default_korean_corpus())
        for p in (
            "data/corpora/ko_wiki_combined.txt",
            "data/corpora/ko_wiki_lines.txt",
            "data/synthetic/subset_ko/labels.tsv",
        ):
            fp = Path(p)
            if fp.exists():
                try:
                    for ln in fp.read_text(encoding="utf-8").splitlines():
                        if "\t" in ln:
                            ln = ln.split("\t", 1)[1]
                        if ln.strip():
                            lines.append(ln)
                except Exception:  # noqa: BLE001
                    pass
        cls._BASE_CACHE = lines
        return lines

    @classmethod
    def _load_ext_only(cls) -> list[str]:
        """Extra lines only present when include_extended=True (i.e. deltas)."""
        if cls._EXT_CACHE is not None:
            return cls._EXT_CACHE
        extras: list[str] = []
        fp = Path("data/corpora/ko_wiki_streamed.txt")
        if fp.exists():
            try:
                extras.extend(fp.read_text(encoding="utf-8").splitlines())
            except Exception:  # noqa: BLE001
                pass
        kp = Path("data/hf/korean-pair/labels.tsv")
        if kp.exists():
            try:
                for ln in kp.read_text(encoding="utf-8").splitlines():
                    if "\t" not in ln:
                        continue
                    t = ln.split("\t", 1)[1].strip()
                    if not t:
                        continue
                    ko = sum(1 for c in t if 0xAC00 <= ord(c) <= 0xD7A3)
                    if ko >= max(3, len(t) * 0.30):
                        extras.append(t)
            except Exception:  # noqa: BLE001
                pass
        cls._EXT_CACHE = extras
        return extras

    def _build_lm(self, include_extended: bool = False) -> CharBigramLM:
        """Fit a char-bigram LM from corpora. Two variants:
            include_extended=False   base  (multi_poly path, _38 optimum)
            include_extended=True    base + korean-pair + ko_wiki_streamed
                                     (single_line path, _41 optimum)

        Base corpus is cached across calls so building both LMs doesn't
        re-read ~5 MB of corpora files twice.
        """
        lines = list(self._load_base_corpus())
        if include_extended:
            lines.extend(self._load_ext_only())
        lm = CharBigramLM(alpha=0.5)
        lm.fit(lines)
        log.info("pipeline.lm_fit n_lines=%d include_extended=%s",
                 len(lines), include_extended)
        return lm

    def _peek_rec_ckpt_meta(self) -> tuple[str | None, list[str] | None]:
        """Peek at the recognizer checkpoint to read its arch + vocab_chars
        BEFORE we instantiate the model. Different tiers ship different
        architectures (lite=svtr_lite dim 192, medical=svtr_medical dim 320);
        without this, build_recognition would instantiate the wrong shape
        and load_state_dict(strict=False) would silently drop most weights."""
        ckpt_path = (
            self.checkpoints_root / self.config.tier / self.config.lang / "rec.pth"
        )
        if not ckpt_path.exists():
            return None, None
        try:
            raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except Exception:
            return None, None
        if not isinstance(raw, dict):
            return None, None
        return raw.get("arch"), raw.get("vocab_chars")

    def _ensure_loaded(self) -> None:
        if self._detector is None:
            det = DBNet(backbone=self.config.detection.backbone).to(self.device).eval()
            self._load_checkpoint_if_present("det", det)
            self._detector = det
        if self._recognizer is None:
            ckpt_arch, ckpt_chars = self._peek_rec_ckpt_meta()
            arch = ckpt_arch or self.config.recognition.arch
            # If the ckpt ships its own vocab, size the model head to match
            # it now so load_state_dict can restore full weights (head is
            # the single largest tensor). Note: ckpt_chars is the BASE
            # char list; Vocab(...).size adds 5 special tokens, and that
            # total is what the head was built against at training time.
            if ckpt_chars:
                vocab_size = Vocab(
                    name=self.config.recognition.vocab_name,
                    chars=ckpt_chars,
                ).size
            else:
                vocab_size = self._vocab.size
            rec = build_recognition(
                arch,
                vocab_size=vocab_size,
                input_height=self.config.recognition.input_height,
            ).to(self.device).eval()
            self._load_checkpoint_if_present("rec", rec)
            # _60: optional INT8 dynamic quantization on CPU.
            # Enable with SPINAI_QUANTIZE=int8 (Railway prod toggle).
            # Quantizes Linear layers only — safe for SVTR attention/FFN
            # which dominate runtime. CPU-only; skipped if running on CUDA.
            import os as _os
            if (_os.environ.get("SPINAI_QUANTIZE", "").lower() == "int8"
                    and str(self.device) == "cpu"):
                try:
                    rec = torch.quantization.quantize_dynamic(
                        rec, {torch.nn.Linear}, dtype=torch.qint8,
                    )
                    log.info("recognizer.quantized_int8 device=cpu")
                except Exception as e:  # noqa: BLE001
                    log.warning("quantize_dynamic_failed: %s — keeping fp32", e)
            self._recognizer = rec
            # _84 (iter 27): optional secondary for ensemble.
            # iter 78: SPINAI_ENSEMBLE_REC_FILE override allows arbitrary
            # secondary ckpt filename (default kept for back-compat with iter 27).
            if _os.environ.get("SPINAI_ENSEMBLE", "") == "1" and self.config.tier == "consumer_v1":
                sec_filename = _os.environ.get(
                    "SPINAI_ENSEMBLE_REC_FILE", "rec_v2d_pre_iter20.pth.bak"
                )
                sec_path = (
                    self.checkpoints_root / self.config.tier / self.config.lang
                    / sec_filename
                )
                if sec_path.exists():
                    try:
                        sec = build_recognition(
                            arch, vocab_size=vocab_size,
                            input_height=self.config.recognition.input_height,
                        ).to(self.device).eval()
                        raw = torch.load(sec_path, map_location=self.device, weights_only=False)
                        state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
                        state = {k.replace("student.", "", 1): v for k, v in state.items()}
                        sec.load_state_dict(state, strict=False)
                        self._recognizer_secondary = sec
                        log.info("ensemble.secondary_loaded path=%s", sec_path)
                    except Exception as e:  # noqa: BLE001
                        log.warning("ensemble.secondary_load_failed %s: %s", sec_path, e)

    # ----- pipeline ------------------------------------------------------

    def _easyocr_fallback(self, img: np.ndarray) -> str | None:
        """iter 57: lazy-init EasyOCR ko+en, run on original crop, return
        joined text in geometric reading order. Returns None on init failure
        (callers fall back to SPINAI text). Reader cached after first call.

        iter 134: convert RGB→BGR before readtext. Our internal `img` is RGB
        (PIL Image.open(...).convert("RGB") → np.asarray, see _to_numpy).
        EasyOCR's reformat_input treats a 3-channel ndarray as BGR — its
        grayscale conversion uses cv2.COLOR_BGR2GRAY (`0.114B+0.587G+0.299R`)
        and the recognizer reads from that grayscale. Feeding RGB swaps the R/B
        coefficients in the grayscale used by recognition. Pre-converting to
        BGR makes EasyOCR's channel assumptions consistent with our input.
        """
        if self._easyocr_reader is None:
            try:
                import easyocr
                self._easyocr_reader = easyocr.Reader(["ko", "en"], verbose=False)
                log.info("hybrid.easyocr_reader_loaded")
            except Exception as e:  # noqa: BLE001
                log.warning("hybrid.easyocr_init_failed %s", e)
                self._easyocr_reader = False  # poison sentinel — don't retry
                return None
        if self._easyocr_reader is False:
            return None
        try:
            if img.ndim == 3 and img.shape[2] == 3:
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                bgr = img
            raw = self._easyocr_reader.readtext(bgr, detail=1)
            raw = sorted(raw, key=lambda it: (min(p[1] for p in it[0]),
                                                 min(p[0] for p in it[0])))
            return " ".join(t for _, t, _ in raw).strip()
        except Exception as e:  # noqa: BLE001
            log.warning("hybrid.easyocr_call_failed %s", e)
            return None

    def _easyocr_readtext_multi(self, img: np.ndarray) -> list[tuple[np.ndarray, str, float]]:
        """EasyOCR readtext returning individual lines as (polygon, text, conf)."""
        if self._easyocr_reader is None:
            try:
                import easyocr
                self._easyocr_reader = easyocr.Reader(["ko", "en"], verbose=False)
            except Exception as e:
                log.warning("hybrid.easyocr_init_failed %s", e)
                self._easyocr_reader = False
                return []
        if self._easyocr_reader is False:
            return []
        try:
            if img.ndim == 3 and img.shape[2] == 3:
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                bgr = img
            raw = self._easyocr_reader.readtext(bgr, detail=1)
            results = []
            for bbox, text, conf in raw:
                if not text.strip():
                    continue
                poly = np.array(bbox, dtype=np.float32)
                results.append((poly, text.strip(), float(conf)))
            results.sort(key=lambda x: (x[0][:, 1].min(), x[0][:, 0].min()))
            return results
        except Exception as e:
            log.warning("hybrid.easyocr_multi_failed %s", e)
            return []

    def _vlm_readtext_multi(self, img: np.ndarray) -> list[tuple[np.ndarray, str, float]]:
        """2026-06-10 accuracy-tier: EasyOCR detection + PaddleOCR-VL recognition.
        Measured win: scene char-recall 94% (vs product 88%) / clean-doc 90% (vs
        70%). VL runs as an external GPU microservice (scripts/vl_service.py); the
        client degrades to the EasyOCR text per-crop when the service is
        unreachable, so this never scores below ``easyocr_only``.
        """
        multi = self._easyocr_readtext_multi(img)
        if not multi:
            return []
        from spinai_ocr.inference.vl_client import vl_ocr, vl_healthy
        if not vl_healthy():
            log.warning("hybrid.vlm_service_down -> degrade to easyocr text")
            return multi
        from PIL import Image as _Image
        out = []
        for poly, eo_text, conf in multi:
            xs = poly[:, 0]; ys = poly[:, 1]
            x0, y0 = int(max(0, xs.min())), int(max(0, ys.min()))
            x1, y1 = int(xs.max()), int(ys.max())
            if x1 - x0 < 4 or y1 - y0 < 4:
                out.append((poly, eo_text, conf)); continue
            crop = _Image.fromarray(img[y0:y1, x0:x1])
            out.append((poly, vl_ocr(crop, fallback=eo_text), conf))
        return out

    # iter 131: per-mode routing presets.
    # (line_thresh, cov_thresh, cov_signal). cov_thresh=None disables
    # whole-image fallback (fast mode = per-line only).
    # iter 145: `easyocr_only` is special-cased in `_apply_hybrid_routing`
    # — bypasses v031 entirely and returns EasyOCR readtext on whole image.
    # The preset values below are sentinels (line_thresh / cov_thresh
    # never read for this mode) but we enroll the mode in the dict so
    # `_resolve_routing_params` recognizes it.
    _ROUTING_PRESETS: dict[str, tuple[float, float | None, str]] = {
        "fast":          (0.70, None, "min_conf"),
        "balanced":      (0.70, 0.65, "min_conf"),
        "accurate":      (0.90, 0.80, "min_conf"),
        "easyocr_only":  (0.0,  None, "min_conf"),  # sentinel — see _apply_hybrid_routing
        "vlm":           (0.0,  None, "min_conf"),  # sentinel — accuracy tier, EasyOCR det + PaddleOCR-VL rec
    }

    def _easyocr_recognize_line_crop(
        self,
        img: np.ndarray,
        polygon: np.ndarray,
        pad_ratio: float = 0.10,
        min_pad_px: int = 4,
    ) -> str | None:
        """iter 131 (productionize iter 129 pattern): EasyOCR recognize-direct
        on a padded axis-aligned bbox of the polygon. ~3.5× faster than
        readtext on per-line crops because CRAFT detection is skipped.
        Returns None on init failure or empty output."""
        if self._easyocr_reader is None:
            try:
                import easyocr  # type: ignore
                self._easyocr_reader = easyocr.Reader(["ko", "en"], verbose=False)
                log.info("hybrid.easyocr_reader_loaded")
            except Exception as e:  # noqa: BLE001
                log.warning("hybrid.easyocr_init_failed %s", e)
                self._easyocr_reader = False  # poison sentinel
                return None
        if self._easyocr_reader is False:
            return None
        H, W = img.shape[:2]
        xs = polygon[:, 0]
        ys = polygon[:, 1]
        x0, y0 = float(xs.min()), float(ys.min())
        x1, y1 = float(xs.max()), float(ys.max())
        box_w = max(1.0, x1 - x0)
        box_h = max(1.0, y1 - y0)
        pad_x = max(min_pad_px, int(box_w * pad_ratio))
        pad_y = max(min_pad_px, int(box_h * pad_ratio))
        x0i = max(0, int(x0) - pad_x)
        y0i = max(0, int(y0) - pad_y)
        x1i = min(W, int(x1) + pad_x)
        y1i = min(H, int(y1) + pad_y)
        if x1i <= x0i or y1i <= y0i:
            return None
        crop = img[y0i:y1i, x0i:x1i]
        if crop.size == 0:
            return None
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
            ch, cw = gray.shape[:2]
            out = self._easyocr_reader.recognize(
                gray, [[0, cw, 0, ch]], [], detail=1
            )
            if not out:
                return None
            return str(out[0][1]).strip() or None
        except Exception as e:  # noqa: BLE001
            log.warning("hybrid.easyocr_recognize_failed %s", e)
            return None

    def _resolve_routing_params(
        self,
    ) -> tuple[str, float, float | None, str] | None:
        """Returns (mode, line_thresh, cov_thresh, cov_signal) when routing is
        enabled, else None. Per-mode preset is the source of defaults; explicit
        user overrides on RecognitionConfig take priority."""
        mode = self.config.recognition.routing_mode
        if mode not in self._ROUTING_PRESETS:
            return None
        p_lt, p_cv, p_cs = self._ROUTING_PRESETS[mode]
        rc = self.config.recognition
        line_thresh = rc.routing_line_thresh if rc.routing_line_thresh is not None else p_lt
        cov_thresh = rc.routing_cov_thresh if rc.routing_cov_thresh is not None else p_cv
        cov_signal = rc.routing_cov_signal or p_cs
        return mode, float(line_thresh), (None if cov_thresh is None else float(cov_thresh)), cov_signal

    # iter 164: image-level subtitle-style classifier for domain-aware routing.
    # Two independent signals that broadcast subtitles share but print/book
    # imagery does not. EITHER fires the subtitle classification:
    #   (1) HSV saturation p90 ≥ 60 — vibrant background (real colour caps)
    #   (2) Otsu outline-ring density ≥ 0.12 — black stroke around glyphs
    # Calibration on iter 164 fixtures (dialogue n=220 + book_holdout n=200,
    # both grayscale-processed in this corpus): sat signal degenerate (S=0),
    # ring at 0.12 separates dialogue 39.5% from book 2.0% — Pareto-favourable
    # trade-off (dialogue ~13% rel CER drop projected, book ~1% noise).
    # Returns False on tiny / non-RGB inputs so legacy routing runs.
    _SUBTITLE_SAT_P90_THRESH: float = 60.0       # 0-255 scale
    _SUBTITLE_RING_DENSITY_THRESH: float = 0.12  # ring px / total px (iter 164 calibration)

    def _is_subtitle_style(self, img: np.ndarray) -> bool:
        if img is None or img.ndim != 3 or img.shape[2] != 3:
            return False
        h, w = img.shape[:2]
        if h < 16 or w < 32:
            return False
        try:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            sat_p90 = float(np.percentile(hsv[..., 1], 90))
            if sat_p90 >= self._SUBTITLE_SAT_P90_THRESH:
                return True
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if mask.mean() > 127:  # text expected to be the smaller side
                mask = cv2.bitwise_not(mask)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            dilated = cv2.dilate(mask, kernel, iterations=1)
            ring = cv2.subtract(dilated, mask)
            ring_density = float(ring.mean()) / 255.0
            return ring_density >= self._SUBTITLE_RING_DENSITY_THRESH
        except Exception as e:  # noqa: BLE001
            log.warning("hybrid.domain_classify_failed %s", e)
            return False

    def _detect_craft(self, img: np.ndarray) -> list[np.ndarray]:
        """Use EasyOCR's CRAFT detector, merging word boxes into lines."""
        if not hasattr(self, "_craft_reader"):
            try:
                import easyocr
                self._craft_reader = easyocr.Reader(
                    [self.config.lang], gpu=self.device.startswith("cuda"),
                    recognizer=False,
                )
            except ImportError:
                log.warning("detect.craft_unavailable fallback=dbnet")
                self._craft_reader = False
        if self._craft_reader is False:
            det_input, scale = _resize_keep_aspect(img, self.config.detection.input_size)
            det_t = _to_tensor(det_input, self.device)
            det_out = self._detector(det_t)
            prob_map = det_out["prob"][0, 0].cpu().numpy()
            polygons = self._post_det.extract_polygons(prob_map, (det_input.shape[0], det_input.shape[1]))
            return [p / scale for p in polygons]
        try:
            horizontal, _ = self._craft_reader.detect(img)
        except Exception as e:
            log.warning("detect.craft_failed fallback=dbnet: %s", e)
            self._craft_reader = False
            det_input, scale = _resize_keep_aspect(img, self.config.detection.input_size)
            det_t = _to_tensor(det_input, self.device)
            det_out = self._detector(det_t)
            prob_map = det_out["prob"][0, 0].cpu().numpy()
            polygons = self._post_det.extract_polygons(prob_map, (det_input.shape[0], det_input.shape[1]))
            return [p / scale for p in polygons]
        if not horizontal or not horizontal[0]:
            return []
        words = []
        for box in horizontal[0]:
            x_min, x_max, y_min, y_max = box
            words.append((float(x_min), float(y_min), float(x_max), float(y_max)))
        words.sort(key=lambda b: (b[1], b[0]))
        lines: list[list[tuple]] = []
        for w in words:
            merged = False
            for line in lines:
                rep = line[-1]
                rh = rep[3] - rep[1]
                wh = w[3] - w[1]
                min_h = min(rh, wh)
                y_overlap = max(0, min(rep[3], w[3]) - max(rep[1], w[1]))
                if min_h > 0 and y_overlap / min_h >= 0.4:
                    line.append(w)
                    merged = True
                    break
            if not merged:
                lines.append([w])
        polygons = []
        for line in lines:
            x0 = min(b[0] for b in line)
            y0 = min(b[1] for b in line)
            x1 = max(b[2] for b in line)
            y1 = max(b[3] for b in line)
            poly = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
            polygons.append(poly)
        return polygons

    def _apply_hybrid_routing(
        self,
        img: np.ndarray,
        polygons: list[np.ndarray],
        texts: list[str],
        confidences: np.ndarray,
        min_char_confs: np.ndarray,
    ) -> tuple[list[np.ndarray], list[str], np.ndarray, np.ndarray]:
        """iter 131: productionize three-mode hybrid routing.
        Pareto modes characterized in iter 128/129/130:
          fast     — per-line replace (iter129 line_conf<0.70)
          balanced — per-line + whole-image fallback when min_conf<0.65 (iter130)
          accurate — line_conf<0.90 + whole-image when min_conf<0.80 (iter130)

        Returns possibly-replaced (polygons, texts, confidences, min_char_confs).
        Whole-image fallback collapses output to a single full-image bbox
        (matches iter130 bench contract). Per-line fallback preserves bboxes.
        EasyOCR import failure → returns inputs unchanged.
        """
        params = self._resolve_routing_params()
        if params is None:
            return polygons, texts, confidences, min_char_confs
        mode, line_thresh, cov_thresh, cov_signal = params

        # iter 164: image-level domain-aware override. When opted in via
        # routing_domain_aware=True and mode is balanced/accurate, classify
        # the image; broadcast-subtitle-style images bypass v031 and route
        # straight to EasyOCR readtext (same effect as easyocr_only but
        # per-image rather than per-call).
        if (self.config.recognition.routing_domain_aware
                and mode in ("balanced", "accurate")
                and self._is_subtitle_style(img)):
            whole = self._easyocr_fallback(img)
            if whole:
                H, W = img.shape[:2]
                poly_full = np.array([[0.0, 0.0], [float(W), 0.0],
                                      [float(W), float(H)], [0.0, float(H)]],
                                     dtype=np.float32)
                log.info("hybrid.routing_domain_aware mode=%s -> easyocr_only", mode)
                return ([poly_full], [whole],
                        np.array([1.0], dtype=np.float32),
                        np.array([1.0], dtype=np.float32))

        # iter 145: easyocr_only — bypass v031 entirely. Strict winner on
        # iter 143 real-photo bench (CER 22.87% vs accurate 26.11%, vs v031
        # None 47.80%). Trade-off: no SPINAI hybrid value-prop; pure
        # EasyOCR-as-engine. Use case = accuracy-first opt-in.
        # Graceful degrade: if EasyOCR init fails, return v031 inputs
        # unchanged so the pipeline does not crash.
        if mode == "easyocr_only":
            multi = self._easyocr_readtext_multi(img)
            if multi:
                r_polys = [r[0] for r in multi]
                r_texts = [r[1] for r in multi]
                r_confs = np.array([r[2] for r in multi], dtype=np.float32)
                r_mcc = np.array([r[2] for r in multi], dtype=np.float32)
                log.info("hybrid.routing_easyocr_only mode=%s n_lines=%d", mode, len(multi))
                return r_polys, r_texts, r_confs, r_mcc
            return polygons, texts, confidences, min_char_confs

        # 2026-06-10 vlm — accuracy tier: EasyOCR detection + PaddleOCR-VL
        # recognition (external GPU service). Degrades to easyocr_only text per
        # crop when the service is down. Opt-in; never a tier default.
        if mode == "vlm":
            multi = self._vlm_readtext_multi(img)
            if multi:
                r_polys = [r[0] for r in multi]
                r_texts = [r[1] for r in multi]
                r_confs = np.array([r[2] for r in multi], dtype=np.float32)
                r_mcc = np.array([r[2] for r in multi], dtype=np.float32)
                log.info("hybrid.routing_vlm mode=%s n_lines=%d", mode, len(multi))
                return r_polys, r_texts, r_confs, r_mcc
            return polygons, texts, confidences, min_char_confs

        # 1. Whole-image fallback (balanced / accurate modes)
        if cov_thresh is not None:
            valid_confs = [float(c) for t, c in zip(texts, confidences) if t.strip()]
            if not valid_confs:
                cov_value = -1.0  # zero-line case → always escalate
            elif cov_signal == "mean_conf":
                cov_value = sum(valid_confs) / len(valid_confs)
            else:
                cov_value = min(valid_confs)
            if cov_value < cov_thresh:
                whole = self._easyocr_fallback(img)
                if whole:
                    H, W = img.shape[:2]
                    poly_full = np.array([[0.0, 0.0], [float(W), 0.0],
                                          [float(W), float(H)], [0.0, float(H)]],
                                         dtype=np.float32)
                    log.info("hybrid.routing_whole_fallback mode=%s cov=%s value=%.3f thresh=%.3f",
                             mode, cov_signal, cov_value, cov_thresh)
                    new_conf = max(0.0, cov_value) if cov_value >= 0 else 0.0
                    return ([poly_full], [whole],
                            np.array([new_conf], dtype=np.float32),
                            np.array([new_conf], dtype=np.float32))

        # 2. Per-line replace via EasyOCR recognize-direct
        new_texts = list(texts)
        new_confs = np.array(confidences, dtype=np.float32, copy=True)
        n_routed = 0
        for i, (t, conf, poly) in enumerate(zip(texts, confidences, polygons)):
            if not t.strip():
                continue
            if float(conf) >= line_thresh:
                continue
            replaced = self._easyocr_recognize_line_crop(img, poly)
            if not replaced:
                continue
            new_texts[i] = replaced
            new_confs[i] = 1.0  # routed → engine asserts confident
            n_routed += 1
        if n_routed:
            log.info("hybrid.routing_per_line mode=%s thresh=%.3f n_routed=%d/%d",
                     mode, line_thresh, n_routed, len(texts))
        return polygons, new_texts, new_confs, min_char_confs

    def _recognize_logits(self, rec_t: torch.Tensor) -> torch.Tensor:
        """Primary + optional secondary ensemble. Returns logits [B, T, V].
        When only primary exists, equivalent to `self._recognizer(rec_t)`."""
        logits = self._recognizer(rec_t)
        if self._recognizer_secondary is not None:
            # Average in softmax-probability space (same-scale combination);
            # convert back to log-prob via log. iter 27.
            p1 = F.softmax(logits.float(), dim=-1)
            p2 = F.softmax(self._recognizer_secondary(rec_t).float(), dim=-1)
            p = (p1 + p2) * 0.5
            logits = torch.log(p.clamp_min(1e-8))
        return logits

    @torch.inference_mode()
    def __call__(
        self,
        image: ImageLike,
        single_line: bool | None = None,
        decode_mode: str | None = None,
    ) -> OCRResult:
        img = _to_numpy(image)
        H, W = img.shape[:2]
        # Reject images too small to contain any legible text. The recognizer
        # was trained at input_height=48 with width ≥ ~20; anything smaller
        # gets upscaled into noise and the model happily outputs hallucinated
        # characters with ~1.0 confidence. Fail closed instead.
        if H < 8 or W < 8:
            log.info("pipeline.done lines=0 reason=image_too_small H=%d W=%d", H, W)
            return OCRResult(lines=[], image_width=W, image_height=H)
        self._ensure_loaded()
        # Per-call decode_mode override — falls back to config default.
        # Keeps concurrent threadpool requests isolated from each other.
        effective_decode = decode_mode or self.config.decode_mode

        # Auto-detect single-line images: very short-aspect (h/w < 0.2) OR
        # caller passed single_line=True. Skip detection and recognize whole
        # image as one crop — gives accurate output on uploaded line crops
        # where detection would otherwise find spurious sub-polygons.
        # iter 88: track whether the single_line path was caller-explicit
        # vs auto-detected. If auto-detected AND the single_line decode
        # returns empty, we should retry via det+rec instead of returning
        # empty — most worst-30 empty_output failures are auto-classified
        # short-aspect images that nonetheless contain multiple text rows
        # (broken-Hangul GT or unusual aspect ratios). Caller-explicit
        # single_line=True is respected as-is (caller knows the input is
        # a line crop and det+rec would over-segment).
        _single_line_auto = single_line is None
        if single_line is None:
            single_line = H <= 96 or (H / max(W, 1)) < 0.25
        if single_line:
            with log_span("pipeline.single_line"):
                # Resize height-preserving to target_h, then cap width at 320
                # (model was trained with max_width=320; beyond that the CNN
                # sees an aspect it never saw → recognition degrades).
                #
                # _126 (iter 71): tested longwidth sliding-window for true_w>320
                # rows (62/262 truncate) to address held-out CER stagnation.
                # NEG: overlap-merge double-counts decoded text where
                # tile boundaries don't produce identical char strings, blew
                # CER 0.265 → 0.946. SVTR's 320-cap path is surprisingly robust
                # (95-char row: CER 0.105 even with 7x compression). Diagnostic
                # also showed short lines (len<=15, CER 0.39) — not long lines
                # (CER 0.22-0.26) — drive stagnation. See _126 for full record.
                ar = W / max(H, 1)
                base_h = self.config.recognition.input_height
                if ar > 8:
                    target_h = max(base_h // 2, 24)
                elif ar > 4:
                    target_h = max(int(base_h * 0.75), 32)
                else:
                    target_h = base_h
                scale = target_h / max(H, 1)
                new_w = min(int(W * scale), self.config.recognition.inference_max_width)
                whole_crop = cv2.resize(img, (max(new_w, 8), target_h), interpolation=cv2.INTER_LINEAR)
                rec_t, _ = _recognition_tensor([whole_crop], self.device)
                logits = self._recognize_logits(rec_t)
                log_probs_t = F.log_softmax(logits.float(), dim=-1)
                top1_logp = log_probs_t.max(dim=2).values  # [1, T]
                conf = float(top1_logp.mean().exp().cpu())
                # _80 (iter 24): per-step min top-1 softmax prob. Single weakest
                # step across the line — better UX signal than geometric mean
                # for "which character looks suspicious".
                min_char_conf = float(top1_logp.min().exp().cpu())
                # _74 / _78: apply per-tier calibration power (1.0 = identity).
                cp = self.config.recognition.conf_power
                if cp != 1.0:
                    conf = max(0.0, min(1.0, conf ** cp))
                    min_char_conf = max(0.0, min(1.0, min_char_conf ** cp))
                log_probs_np = log_probs_t.cpu().numpy()
                if effective_decode == "greedy":
                    texts = decode_batch(log_probs_np, self._vocab, mode="greedy",
                                         blank_penalty=self.config.recognition.blank_penalty)
                else:
                    if self._lm_single_line is None:
                        self._lm_single_line = self._build_lm(include_extended=True)
                    texts = decode_batch(
                        log_probs_np, self._vocab, mode="beam_lm",
                        beam_cfg=BeamConfig(beam_width=BEAM_WIDTH_SINGLE_LINE, topk_per_step=BEAM_TOPK),
                        lm=self._lm_single_line, lm_alpha=LM_ALPHA_SINGLE_LINE,
                        is_log_probs=True,
                        blank_penalty=self.config.recognition.blank_penalty,
                    )
                # iter 132: routing_mode unification — when set,
                # _apply_hybrid_routing supersedes any single_line fallback.
                # iter 141: legacy `easyocr_fallback_threshold` else-branch
                # removed. consumer_v1 default is now routing_mode="balanced"
                # (set in __init__ via _TIER_DEFAULT_ROUTING_MODE). The
                # iter 130 OOD bench (n=500 det_ko_v2) showed per-line
                # routing Pareto-dominates the single-threshold fallback.
                if self.config.recognition.routing_mode is not None:
                    sl_poly = np.array(
                        [[0.0, 0.0], [float(W), 0.0], [float(W), float(H)], [0.0, float(H)]],
                        dtype=np.float32,
                    )
                    with log_span(
                        "pipeline.hybrid_routing",
                        mode=self.config.recognition.routing_mode,
                        path="single_line",
                    ):
                        r_polys, r_texts, r_confs, r_mcc = self._apply_hybrid_routing(
                            img, [sl_poly], [texts[0]],
                            np.asarray([conf], dtype=np.float32),
                            np.asarray([min_char_conf], dtype=np.float32),
                        )
                    if r_texts and r_texts[0]:
                        rp = r_polys[0]
                        return OCRResult(
                            lines=[OCRLine(
                                text=r_texts[0],
                                bbox=[(float(rp[0][0]), float(rp[0][1])),
                                      (float(rp[1][0]), float(rp[1][1])),
                                      (float(rp[2][0]), float(rp[2][1])),
                                      (float(rp[3][0]), float(rp[3][1]))],
                                confidence=float(r_confs[0]),
                                min_char_conf=float(r_mcc[0]),
                            )],
                            image_width=W, image_height=H,
                        )
                    # routing returned empty → fall through to empty-retry-with-det.
                # iter 88: empty single_line output on auto-detected path
                # → fall through to det+rec rather than returning "". Worst-30
                # audit (n=200) showed 19/30 worst cases were empty_output
                # with conf 0.013-0.26 on long/multi-row images that the H/W
                # heuristic mis-classified as single_line.
                if not texts[0] and _single_line_auto:
                    log.info("single_line.empty_retry_with_det conf=%.3f", conf)
                    # fall through to detection path below
                else:
                    # iter 166: jamo correction on single-line path
                    _sl_text = texts[0]
                    if self.config.recognition.use_jamo_correction and _sl_text:
                        from spinai_ocr.postprocess.jamo_correct import correct_jamo
                        _sl_text = correct_jamo(_sl_text)
                    return OCRResult(
                        lines=[OCRLine(
                            text=_sl_text,
                            bbox=[(0.0, 0.0), (float(W), 0.0), (float(W), float(H)), (0.0, float(H))],
                            confidence=conf,
                            min_char_conf=min_char_conf,
                        )],
                        image_width=W, image_height=H,
                    )
        import time as _time_mod
        _t_pipeline_start = _time_mod.perf_counter()
        log.info("pipeline.start H=%d W=%d lang=%s tier=%s decode=%s",
                 H, W, self.config.lang, self.config.tier, effective_decode,
                 extra={"H": H, "W": W, "lang": self.config.lang, "tier": self.config.tier,
                        "decode": effective_decode, "phase": "pipeline_start"})

        # 1. preprocess
        # iter 91: train-test CLAHE mismatch — augment.py applies CLAHE to
        # only 30% of training samples (AugProb 0.3) but inference was running
        # it 100% on consumer_v1+. Skipping it gives matched-CER −3.3% rel
        # on synth det_ko n=200 (0.0459 → 0.0444), neutral F1, neutral lat
        # at typical synth size, and removes 30ms p50 cost on big-photo
        # (4000×3000) workloads. Caller opt-in CLAHE (?preprocess=clahe in
        # serve/app.py) still runs before pipeline if the user wants it.
        with log_span("pipeline.preprocess", tier=self.config.tier):
            img_proc = img

        # 2. detect
        _t_detect = _time_mod.perf_counter()
        with log_span("pipeline.detect"):
            if self.config.detection.backend == "craft":
                polygons = self._detect_craft(img_proc)
            else:
                det_input, scale = _resize_keep_aspect(img_proc, self.config.detection.input_size)
                det_t = _to_tensor(det_input, self.device)
                det_out = self._detector(det_t)
                prob_map = det_out["prob"][0, 0].cpu().numpy()
                polygons = self._post_det.extract_polygons(prob_map, (det_input.shape[0], det_input.shape[1]))
                polygons = [p / scale for p in polygons]
            MIN_POLY_H = 12.0
            MIN_POLY_W = 16.0
            kept: list[np.ndarray] = []
            for p in polygons:
                if p.shape[0] < 3:
                    continue
                xs, ys = p[:, 0], p[:, 1]
                w = float(xs.max() - xs.min())
                h = float(ys.max() - ys.min())
                if w >= MIN_POLY_W and h >= MIN_POLY_H:
                    kept.append(p)
            n_dropped_micro = len(polygons) - len(kept)
            polygons = kept
            if n_dropped_micro:
                log.debug("detect.filter_micro dropped=%d", n_dropped_micro)
            import os as _os
            if self.config.detection.backend != "craft":
                if _os.environ.get("SPINAI_DISABLE_HMERGE") != "1":
                    polygons = _merge_horizontal_boxes(polygons)
            img_area = float(img.shape[0] * img.shape[1])
            polygons = _split_tall_boxes(img, polygons, img_area=img_area)
            _detect_ms = round((_time_mod.perf_counter() - _t_detect) * 1000, 1)
            log.info("pipeline.detect.done lang=%s boxes=%d elapsed_ms=%.1f",
                     self.config.lang, len(polygons), _detect_ms,
                     extra={"phase": "detect_done", "lang": self.config.lang,
                            "tier": self.config.tier, "n_boxes": len(polygons),
                            "elapsed_ms": _detect_ms})

        # 3. refine
        if polygons:
            with log_span("pipeline.refine", n_polys=len(polygons)):
                polygons = self._refiner.refine(img, polygons)

        if not polygons:
            _total_ms = round((_time_mod.perf_counter() - _t_pipeline_start) * 1000, 1)
            log.info("pipeline.done lang=%s lines=0 reason=no_polygons elapsed_ms=%.1f",
                     self.config.lang, _total_ms,
                     extra={"phase": "pipeline_done", "lang": self.config.lang,
                            "tier": self.config.tier, "n_lines": 0,
                            "reason": "no_polygons", "elapsed_ms": _total_ms})
            return OCRResult(lines=[], image_width=W, image_height=H,
                             timing_ms={"detect_ms": _detect_ms, "total_ms": _total_ms})

        # 4. recognize
        _t_recognize = _time_mod.perf_counter()
        with log_span("pipeline.recognize", n_polys=len(polygons)):
            # iter 113: dilate each tight poly for rec-input crop only. The
            # original tight polygon (`p`) is what we return + F1-score against;
            # the dilated version goes to the recognizer for a comfortable crop.
            # When recognize_extra_ratio=0.0, dilate_for_recognize returns p
            # unchanged (back-compat).
            crops = [_crop_polygon(img, self._post_det.dilate_for_recognize(p),
                                    target_h=self.config.recognition.input_height)
                     for p in polygons]
            # iter 166: AngleClassifier — correct 90/180/270 rotation on each crop
            # before recognition. Loads a tiny CNN (48x192 -> 4-class). Only applied
            # when a trained checkpoint exists at checkpoints/<tier>/<lang>/angle_cls.pth.
            # ROI: catches scan-rotated pages / mobile uploads with wrong orientation.
            crops = self._apply_angle_correction(crops)
            rec_t, rec_widths = _recognition_tensor(crops, self.device)
            if rec_t.shape[0] == 0:
                log.warning("recognize.skipped reason=empty_crops n_polys=%d", len(polygons))
                return OCRResult(lines=[], image_width=W, image_height=H)
            logits = self._recognize_logits(rec_t)
            # Compute log-softmax once on the device; reuse for both the
            # is-text filter and the beam decoder (the decoder would otherwise
            # recompute log-softmax from scratch in NumPy per crop).
            log_probs = F.log_softmax(logits.float(), dim=-1)
            blank_id = self._vocab.blank_id
            blank_logp = log_probs[:, :, blank_id]
            max_nb = (1.0 - blank_logp.exp()).max(dim=1).values.cpu().numpy()
            # _90 (iter 35): tighten max_nb filter 0.70 → 0.80. Held-out
            # tail analysis showed SPINAI over-produces on noisy multi-
            # region pages: single-char fragments ("T", "n", "1") appear
            # as weakly-recognized junk. A crop whose strongest non-blank
            # step still can't clear 0.8 is almost certainly a bad region.
            # Turn off with SPINAI_MAXNB=<float> (e.g. 0.70 = iter 19).
            import os as _os
            _max_nb_thr = float(_os.environ.get("SPINAI_MAXNB", "0.80"))
            keep_mask = max_nb >= _max_nb_thr
            # Per-crop confidence = geometric mean of top-1 prob per step.
            # Captures both "is model peaky?" and "is the peak consistent?".
            # Replaces the previous hard-coded confidence=1.0 placeholder.
            top1_logp = log_probs.max(dim=2).values  # [B, T]
            confidences = top1_logp.mean(dim=1).exp().cpu().numpy()
            # _80 (iter 24): per-crop min top-1 prob (weakest step).
            # iter 103: this is the uncalibrated value; reuse it for low_idx
            # selection below (was previously recomputed at line 938 — same
            # tensor op, same .cpu() sync, pure waste).
            raw_min = top1_logp.min(dim=1).values.exp().cpu().numpy()
            min_char_confs = raw_min
            # _74 / _78: apply per-tier post-hoc calibration.
            cp = self.config.recognition.conf_power
            if cp != 1.0:
                confidences = np.clip(confidences ** cp, 0.0, 1.0)
                # np.clip allocates a new array → raw_min stays uncalibrated.
                min_char_confs = np.clip(raw_min ** cp, 0.0, 1.0)

            # _81 (iter 25) / _82 (iter 26): selective TTA on low-conf crops.
            # Rotation TTA (±2°) broke Korean syllable layout in iter 25 (+30%
            # CER). Scale-only TTA (0.9/1.0/1.1) in iter 26 wins: CER mean
            # 0.261 → 0.248 (−5%), median −10%, p95 −8%, max −6%. +18% latency
            # at p50 only on low-conf crops. Scale-only is now the default;
            # turn off with `SPINAI_DISABLE_TTA=1` or switch mode with
            # `SPINAI_TTA_MODE=rotation` (for non-CJK experiments).
            import os as _os
            tta_enabled = _os.environ.get("SPINAI_DISABLE_TTA") != "1"
            # iter 103: low_idx skips keep_mask=False crops. Their texts get
            # blanked to "" in the post-decode filter regardless, so any TTA
            # work on them is wasted (TTA never refreshes keep_mask itself —
            # current code only refreshes confidences/min_char_confs after
            # TTA, so a TTA-boosted max_nb wouldn't flip the keep decision
            # anyway). Quality is byte-identical to before.
            low_idx = [i for i in range(len(raw_min))
                        if raw_min[i] < 0.75 and bool(keep_mask[i])]
            if tta_enabled and low_idx:
                from spinai_ocr.inference.tta_ensemble import tta_predict, TTAConfig
                # iter 25 에서 rotation 포함 TTA 는 한글 geometry 교란 → NEG.
                # iter 26: scale-only (1.0 / 1.1 / 0.9). 한글 syllable 은
                # size 에 더 robust. 환경변수 `SPINAI_TTA_MODE=rotation` 로
                # rotation 포함 실험도 가능.
                _tta_mode = _os.environ.get("SPINAI_TTA_MODE", "scale")
                if _tta_mode == "rotation":
                    tta_cfg = TTAConfig(scales=(1.0,), rotations_deg=(-2.0, 0.0, 2.0))
                else:  # scale-only (default if TTA enabled)
                    tta_cfg = TTAConfig(scales=(0.9, 1.0, 1.1), rotations_deg=(0.0,))
                tta_x = rec_t[low_idx]
                tta_probs = tta_predict(self._recognizer, tta_x, tta_cfg)
                # Replace those rows in log_probs with log(tta probs). Need
                # to handle shape mismatch (tta returns softmax; we need log).
                tta_logprobs = torch.log(tta_probs.clamp_min(1e-8)).cpu()
                # Pad/truncate time dim to match original log_probs[i]
                T_orig = log_probs.shape[1]
                log_probs_cpu = log_probs.cpu()
                for local_i, global_i in enumerate(low_idx):
                    t_view = tta_logprobs[local_i]
                    if t_view.shape[0] == T_orig:
                        log_probs_cpu[global_i] = t_view
                    elif t_view.shape[0] < T_orig:
                        log_probs_cpu[global_i, :t_view.shape[0]] = t_view
                    else:
                        log_probs_cpu[global_i] = t_view[:T_orig]
                log_probs = log_probs_cpu
                # Refresh calibrated confidences/min_char_conf on TTA rows
                top1_logp_new = log_probs.max(dim=2).values
                new_conf = top1_logp_new.mean(dim=1).exp().numpy()
                new_min = top1_logp_new.min(dim=1).values.exp().numpy()
                for gi in low_idx:
                    c = new_conf[gi]
                    m = new_min[gi]
                    if cp != 1.0:
                        c = min(1.0, max(0.0, c ** cp))
                        m = min(1.0, max(0.0, m ** cp))
                    confidences[gi] = c
                    min_char_confs[gi] = m
                log.info("pipeline.tta_applied n=%d/%d", len(low_idx), len(raw_min),
                          extra={"n_tta": len(low_idx), "n_total": len(raw_min)})

            # iter 166: low-conf crop upscale retry.
            # When a crop's first-pass confidence < 0.5, re-run recognition on a
            # 1.5x upscale (cv2.INTER_CUBIC). Keep whichever result has higher
            # confidence. No EasyOCR involved; pure model-side free gain.
            if self.config.recognition.low_conf_upscale_retry:
                _low_conf_retry_idx = [
                    i for i in range(len(confidences))
                    if confidences[i] < 0.5 and bool(keep_mask[i])
                ]
                if _low_conf_retry_idx:
                    _upscaled_crops = []
                    for _ri in _low_conf_retry_idx:
                        _c = crops[_ri]
                        _uh = max(int(_c.shape[0] * 1.5), 1)
                        _uw = max(int(_c.shape[1] * 1.5), 1)
                        _upscaled_crops.append(
                            cv2.resize(_c, (_uw, _uh), interpolation=cv2.INTER_CUBIC)
                        )
                    _up_rec_t, _up_rec_widths = _recognition_tensor(_upscaled_crops, self.device)
                    _up_logits = self._recognize_logits(_up_rec_t)
                    _up_log_probs = F.log_softmax(_up_logits.float(), dim=-1)
                    _up_top1 = _up_log_probs.max(dim=2).values
                    _up_conf_raw = _up_top1.mean(dim=1).exp().cpu().numpy()
                    _cp = self.config.recognition.conf_power
                    _up_conf = (
                        np.clip(_up_conf_raw ** _cp, 0.0, 1.0) if _cp != 1.0
                        else _up_conf_raw
                    )
                    _n_improved = 0
                    _conf_deltas: list[float] = []
                    for _local_i, _global_i in enumerate(_low_conf_retry_idx):
                        _conf_before = confidences[_global_i]
                        _conf_after  = float(_up_conf[_local_i])
                        _improved = _conf_after > _conf_before
                        log.debug(
                            "upscale_retry.crop idx=%d conf_before=%.3f conf_after=%.3f improved=%s",
                            _global_i, _conf_before, _conf_after, _improved,
                            extra={
                                "phase": "upscale_retry", "crop_idx": _global_i,
                                "conf_before": round(_conf_before, 4),
                                "conf_after": round(_conf_after, 4),
                                "improved": _improved,
                            },
                        )
                        if _improved:
                            # Swap log_probs row with upscaled version
                            _t_up = _up_log_probs[_local_i].cpu()
                            _lp_cpu = log_probs.cpu() if torch.is_tensor(log_probs) else torch.from_numpy(log_probs)
                            T_orig_ = _lp_cpu.shape[1]
                            T_up_   = _t_up.shape[0]
                            if T_up_ >= T_orig_:
                                _lp_cpu[_global_i] = _t_up[:T_orig_]
                            else:
                                _lp_cpu[_global_i, :T_up_] = _t_up
                            log_probs = _lp_cpu
                            confidences[_global_i] = _conf_after
                            _conf_deltas.append(_conf_after - _conf_before)
                            _n_improved += 1
                    _n_tried = len(_low_conf_retry_idx)
                    _n_total = len(confidences)
                    if _n_tried > _n_total * 0.5:
                        log.warning(
                            "upscale_retry.high_retry_rate n_low_conf=%d/%d "
                            "— image quality may be poor",
                            _n_tried, _n_total,
                            extra={
                                "phase": "upscale_retry",
                                "n_low_conf": _n_tried, "n_total": _n_total,
                            },
                        )
                    _avg_delta = round(sum(_conf_deltas) / len(_conf_deltas), 4) if _conf_deltas else 0.0
                    log.info(
                        "pipeline.upscale_retry tried=%d improved=%d avg_conf_delta=%.4f",
                        _n_tried, _n_improved, _avg_delta,
                        extra={
                            "phase": "upscale_retry",
                            "n_tried": _n_tried, "n_improved": _n_improved,
                            "avg_conf_delta": _avg_delta,
                        },
                    )

            log_probs_np = log_probs.cpu().numpy() if torch.is_tensor(log_probs) else np.asarray(log_probs)
            # iter 104: per-row T-slice. _recognition_tensor pads every crop
            # to max(widths). Each row's log_probs[:T_full, :] beyond
            # T_i = w_i / stride corresponds to the white right-pad region;
            # the model is trained to emit blank there, so decoding those
            # steps is wasted work (and CTC beam search occasionally injects
            # padding-region hallucinations). Compute T_i from actual widths
            # and slice each row before decode. Quality byte-tied or improved.
            T_full = log_probs_np.shape[1]
            max_w = max(rec_widths) if rec_widths else 1
            row_T = [max(1, int(w * T_full // max_w)) for w in rec_widths]
            if effective_decode == "greedy":
                # Fast path: ~10× faster decoding; lower quality on long lines.
                texts = decode_batch(log_probs_np, self._vocab, mode="greedy",
                                     blank_penalty=self.config.recognition.blank_penalty,
                                     row_T=row_T)
            else:
                if self._lm_multi_poly is None:
                    self._lm_multi_poly = self._build_lm(include_extended=False)
                texts = decode_batch(
                    log_probs_np, self._vocab, mode="beam_lm",
                    beam_cfg=BeamConfig(beam_width=BEAM_WIDTH_MULTI_POLY, topk_per_step=BEAM_TOPK),
                    lm=self._lm_multi_poly, lm_alpha=LM_ALPHA_MULTI_POLY,
                    is_log_probs=True,
                    blank_penalty=self.config.recognition.blank_penalty,
                    row_T=row_T,
                )

            def _looks_hallucinated(t: str) -> bool:
                s = t.strip()
                if len(s) < 4:
                    return False
                return len(set(s)) / max(len(s), 1) < 0.50

            # _90 (iter 35): also drop very-short outputs (1-2 chars) when
            # the crop's weakest step is weak. Held-out tail showed SPINAI
            # producing single-char garbage ("T", "n", "1", "O") alongside
            # real text, inflating per-page CER. Short outputs from clean
            # crops (>= 0.5 min_char_conf) are kept — think headers like
            # "No" or dates like "5".
            def _is_short_lowconf(t: str, mcc: float) -> bool:
                s = t.strip()
                return 1 <= len(s) <= 2 and mcc < 0.5

            texts = [
                t if (k and not _looks_hallucinated(t) and not _is_short_lowconf(t, mcc)) else ""
                for t, k, mcc in zip(texts, keep_mask, min_char_confs)
            ]
            texts = [_strip_leading_dupe(t) for t in texts]
            # iter 166: rule-based jamo correction (spaces + compat-jamo compose).
            if self.config.recognition.use_jamo_correction:
                from spinai_ocr.postprocess.jamo_correct import correct_jamo
                _texts_before = texts[:]
                texts = [correct_jamo(t) for t in texts]
                _n_jamo_corrected = sum(1 for a, b in zip(_texts_before, texts) if a != b)
                if _n_jamo_corrected:
                    log.info(
                        "pipeline.jamo_correction n_corrected=%d/%d",
                        _n_jamo_corrected, len(texts),
                        extra={
                            "phase": "jamo_correction",
                            "n_corrected": _n_jamo_corrected, "n_total": len(texts),
                        },
                    )
            n_dropped = sum(1 for t in texts if not t)
            if n_dropped:
                log.info("pipeline.filter dropped=%d kept=%d",
                         n_dropped, len(texts) - n_dropped)

            # iter 131: hybrid routing (productionize iter 128/129/130).
            # Disabled by default; enable via RecognitionConfig.routing_mode.
            if self.config.recognition.routing_mode is not None:
                with log_span("pipeline.hybrid_routing",
                              mode=self.config.recognition.routing_mode):
                    polygons, texts, confidences, min_char_confs = (
                        self._apply_hybrid_routing(
                            img, polygons, texts, confidences, min_char_confs,
                        )
                    )

        # 5. optional LLM correction
        if self.config.use_llm_postprocess and any(texts):
            with log_span("pipeline.llm_correct"):
                corrector = self._get_llm_corrector()
                joined = "\n".join(texts)
                corrected = corrector.correct(joined).splitlines()
                if len(corrected) == len(texts):
                    texts = corrected
                else:
                    log.warning("llm.correction line count changed orig=%d new=%d → keeping original",
                                len(texts), len(corrected))

        _recognize_ms = round((_time_mod.perf_counter() - _t_recognize) * 1000, 1)
        _total_ms = round((_time_mod.perf_counter() - _t_pipeline_start) * 1000, 1)
        lines = [
            OCRLine(text=txt, bbox=[tuple(pt) for pt in poly.tolist()],
                    confidence=float(conf),
                    min_char_conf=float(mcc),
                    lang=self.config.lang)
            for txt, poly, conf, mcc in zip(texts, polygons, confidences, min_char_confs)
            if txt.strip()
        ]
        avg_conf = round(float(sum(l.confidence for l in lines) / max(len(lines), 1)), 4)
        log.info(
            "pipeline.done lang=%s lines=%d dropped=%d avg_conf=%.3f "
            "detect_ms=%.1f recognize_ms=%.1f total_ms=%.1f",
            self.config.lang, len(lines), len(texts) - len(lines), avg_conf,
            _detect_ms, _recognize_ms, _total_ms,
            extra={
                "phase": "pipeline_done", "lang": self.config.lang,
                "tier": self.config.tier, "decode": effective_decode,
                "n_lines": len(lines), "n_dropped": len(texts) - len(lines),
                "avg_conf": avg_conf, "detect_ms": _detect_ms,
                "recognize_ms": _recognize_ms, "total_ms": _total_ms,
            },
        )
        return OCRResult(
            lines=lines, image_width=W, image_height=H,
            timing_ms={
                "detect_ms": _detect_ms,
                "recognize_ms": _recognize_ms,
                "total_ms": _total_ms,
            },
        )

    # ----- helpers -------------------------------------------------------

    def _get_llm_corrector(self):
        if self._llm_corrector is None:
            from spinai_ocr.postprocess.llm import CorrectorConfig, LLMCorrector

            self._llm_corrector = LLMCorrector(CorrectorConfig(lang=self.config.lang))
        return self._llm_corrector

    def _apply_angle_correction(self, crops: list) -> list:
        """iter 166: rotate each crop to upright orientation using AngleClassifier.

        Loads from checkpoints/<tier>/<lang>/angle_cls.pth when present.
        If no checkpoint exists, returns crops unchanged (zero latency cost).
        Skips correction when predicted angle is 0 (most common case).
        """
        import os as _os
        # Lazy-load angle classifier checkpoint (tiny CNN)
        if self._angle_classifier is None:
            ckpt_path = (
                self.checkpoints_root / self.config.tier / self.config.lang / "angle_cls.pth"
            )
            if ckpt_path.exists():
                from spinai_ocr.models.angle_cls import AngleClassifier
                _ac = AngleClassifier(num_classes=4)
                raw = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
                _ac.load_state_dict(state, strict=False)
                _ac.eval()
                _ac.to(self.device)
                self._angle_classifier = _ac
                log.info("angle_cls.loaded path=%s", ckpt_path)
            else:
                # No checkpoint: mark as disabled with sentinel False
                log.debug(
                    "angle_cls.ckpt_not_found path=%s — correction disabled", ckpt_path,
                    extra={"phase": "angle_cls", "ckpt_path": str(ckpt_path)},
                )
                self._angle_classifier = False  # type: ignore[assignment]

        if not self._angle_classifier:
            return crops

        from spinai_ocr.models.angle_cls import predict_angle, rotate_to_upright
        corrected = []
        n_rotated = 0
        angle_dist: dict[int, int] = {0: 0, 90: 0, 180: 0, 270: 0}
        for i, crop in enumerate(crops):
            # Build [1,3,H,W] float tensor in [0,1]
            _h, _w = crop.shape[:2]
            _t = torch.from_numpy(
                crop.transpose(2, 0, 1).astype("float32") / 255.0
            ).unsqueeze(0).to(self.device)
            try:
                angle = predict_angle(self._angle_classifier, _t)
            except Exception as _e:
                log.warning(
                    "angle_cls.predict_failed crop=%d: %s — skipping rotation", i, _e,
                    extra={"phase": "angle_cls", "crop_idx": i, "error": str(_e)},
                )
                corrected.append(crop)
                continue
            angle_dist[angle] = angle_dist.get(angle, 0) + 1
            if angle != 0:
                crop = rotate_to_upright(crop, angle)
                n_rotated += 1
            corrected.append(crop)
        log.debug(
            "angle_cls.corrected n_rotated=%d/%d dist=%s",
            n_rotated, len(crops), angle_dist,
            extra={
                "phase": "angle_cls", "n_rotated": n_rotated,
                "n_total": len(crops), "angle_dist": angle_dist,
            },
        )
        return corrected
