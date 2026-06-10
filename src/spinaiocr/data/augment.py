"""Augmentations.

Borrowed from the medical imaging pipeline and adapted for OCR:
- CLAHE       → low-contrast scans / faxes (analogous to X-ray)
- Speckle     → low-quality camera / phone / photocopy
- CopyPaste   → rare character / rare word instance boosting
- plus OCR-specific: motion blur, JPEG compression, random rotate, perspective
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Iterable

import cv2
import numpy as np


Image = np.ndarray  # HxWx3 uint8


# ---------------------------------------------------------------------------
# Basic photometric
# ---------------------------------------------------------------------------


def clahe(img: Image, clip_limit: float = 2.0, tile: int = 8) -> Image:
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    tool = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    l = tool.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)


def speckle(img: Image, sigma: float = 0.08) -> Image:
    noise = np.random.randn(*img.shape).astype(np.float32) * sigma
    out = img.astype(np.float32) / 255.0
    out = out + out * noise
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def jpeg_compress(img: Image, quality_range: tuple[int, int] = (30, 90)) -> Image:
    q = random.randint(*quality_range)
    _, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)


def motion_blur(img: Image, kernel_range: tuple[int, int] = (3, 9)) -> Image:
    k = random.randrange(kernel_range[0], kernel_range[1], 2)
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0 / k
    angle = random.uniform(0, 180)
    M = cv2.getRotationMatrix2D((k / 2, k / 2), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    return cv2.filter2D(img, -1, kernel)


# ---------------------------------------------------------------------------
# Geometric (single image; polygon-aware variants live with detection dataset)
# ---------------------------------------------------------------------------


def random_rotate(img: Image, max_deg: float = 3.0) -> Image:
    h, w = img.shape[:2]
    angle = random.uniform(-max_deg, max_deg)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))


def stroke_width_jitter(img: Image, max_iter: int = 1) -> Image:
    """Randomly erode/dilate to thicken or thin glyph strokes.

    Black-on-white crops: dilate on the inverted image = thicken strokes;
    erode = thin. Applied in intensity space so the effect is symmetric.
    Training signal for decorative fonts (Brush, HiMelody, Gaegu) that
    have variable stroke width the model otherwise never sees.
    _76 (iter 20) addition.
    """
    if max_iter < 1:
        return img
    n = random.randint(1, max_iter)
    mode = random.choice(["thicken", "thin", "none"])
    if mode == "none":
        return img
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if mode == "thicken":
        out_gray = cv2.erode(gray, kernel, iterations=n)  # darker pixels expand
    else:
        out_gray = cv2.dilate(gray, kernel, iterations=n)  # white eats ink
    return cv2.cvtColor(out_gray, cv2.COLOR_GRAY2RGB)


def dilate_pad(img: Image, max_h_ratio: float = 0.25, max_w_ratio: float = 0.06) -> Image:
    """Pad crop with sampled background to simulate test-time pyclipper dilation.

    iter 113 ships det u=2.0 + recognize_extra_ratio=0.4 (chained dilation).
    Effect on crop: pyclipper offset distance ≈ area*e/perim → for H=48 and
    aspect-asymptotic case ~H*e/2 = 9.6 px vertical and proportionally smaller
    horizontal margin. Rec was trained on tight (un-dilated) crops, causing
    a +59% rel mc cost on the main det_ko fixture (iter 113-117 chain).

    iter 118: train rec to be invariant to this padding distribution by
    randomly extending the crop with corner-sampled background. Vertical
    range ~25% H ≈ 12 px (covers test-time 9.6 px + jitter); horizontal
    range ~6% W (smaller because W >> H typically).

    p_fail ≥ 50%: rec may overfit padding (false-positive blank predictions
    on non-padded test crops) or fail to adapt (mc unchanged).
    """
    h, w = img.shape[:2]
    pad_top = random.randint(0, max(1, int(h * max_h_ratio)))
    pad_bot = random.randint(0, max(1, int(h * max_h_ratio)))
    pad_l = random.randint(0, max(1, int(w * max_w_ratio)))
    pad_r = random.randint(0, max(1, int(w * max_w_ratio)))
    if pad_top + pad_bot + pad_l + pad_r == 0:
        return img
    # Sample background from 4 corners (more realistic than pure white;
    # mirrors how real dilated regions sample from source-image neighbourhood)
    corners = np.stack([img[0, 0], img[0, -1], img[-1, 0], img[-1, -1]])
    bg = corners.mean(axis=0).astype(np.uint8).tolist()
    padded = cv2.copyMakeBorder(
        img, pad_top, pad_bot, pad_l, pad_r,
        cv2.BORDER_CONSTANT, value=bg,
    )
    # Resize back to original (h, w). The text now occupies a smaller
    # fraction of the crop — this is precisely the test-time effect of
    # pyclipper dilation followed by rec input-resize to height=48.
    return cv2.resize(padded, (w, h), interpolation=cv2.INTER_LINEAR)


def partial_occlusion(
    img: Image,
    n_patches_range: tuple[int, int] = (1, 2),
    patch_w_ratio: tuple[float, float] = (0.04, 0.10),
    patch_h_ratio: tuple[float, float] = (0.6, 1.0),
) -> Image:
    """Drop 1-2 near-full-height patches simulating smudge / partial occlusion.

    iter 121: long-text crops disproportionately suffer from per-char occlusion
    (more chars → higher chance ≥1 is masked in real-world capture). Patch
    fill-color samples from corner pixels (matches dilate_pad bg style).
    """
    h, w = img.shape[:2]
    n = random.randint(*n_patches_range)
    out = img.copy()
    corners = np.stack([img[0, 0], img[0, -1], img[-1, 0], img[-1, -1]])
    bg = corners.mean(axis=0).astype(np.uint8)
    for _ in range(n):
        pw = max(2, int(w * random.uniform(*patch_w_ratio)))
        ph = max(2, int(h * random.uniform(*patch_h_ratio)))
        x = random.randint(0, max(0, w - pw))
        y = random.randint(0, max(0, h - ph))
        out[y : y + ph, x : x + pw] = bg
    return out


def random_perspective(img: Image, strength: float = 0.02) -> Image:
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jitter = strength * np.array([w, h])
    dst = src + np.random.uniform(-1, 1, src.shape) * jitter
    M = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    return cv2.warpPerspective(img, M, (w, h), borderValue=(255, 255, 255))


def subtitle_overlay(
    img: Image,
    p_outline: float = 0.9,
    p_shadow: float = 0.6,
    p_gradient: float = 0.5,
) -> Image:
    """Korean drama subtitle visual style transform.

    iter 161 confirmed dialogue raw 64% gap is REAL distribution mismatch
    on subtitle imagery, not vocab/fixture noise. Subtitles in TV/film:
      - white (or pale yellow) text body
      - black (or dark) outline stroke
      - subtle drop shadow
      - vibrant or gradient background

    Synthetic crops are dark-text-on-white. This transform extracts the
    text mask via threshold, repaints text=white + outline=black + drop
    shadow, and replaces background with vibrant solid or gradient color.
    Result trains rec to handle inverted polarity + outline ring artifact
    + non-uniform bg.

    iter 162: applied at p≈0.12 alongside real_photo aug to push v040
    toward subtitle-style invariance without dominating training dist.

    p_fail ~50%: synthetic outline thickness/shadow may not match real
    drama subtitles closely enough; or polarity-inversion may regress
    book domain (where text genuinely is dark-on-light).
    """
    h, w = img.shape[:2]
    if h < 10 or w < 10:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Otsu threshold to robustly extract text mask. Assume text is darker
    # than bg (true for synthetic and most book real-photos); invert mask
    # if text turns out to be lighter (bright-on-dark synthetic cases).
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if mask.mean() > 127:
        mask = cv2.bitwise_not(mask)

    # Rebuild background.
    if random.random() < p_gradient:
        c1 = np.array([random.randint(20, 230) for _ in range(3)], dtype=np.float32)
        c2 = np.array([random.randint(20, 230) for _ in range(3)], dtype=np.float32)
        ramp = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
        bg = (c1[None, None, :] * (1.0 - ramp) + c2[None, None, :] * ramp).astype(np.uint8)
        bg = np.broadcast_to(bg, (h, w, 3)).copy()
    else:
        c = np.array([random.randint(40, 230) for _ in range(3)], dtype=np.uint8)
        bg = np.full((h, w, 3), c, dtype=np.uint8)

    out = bg.copy()

    # Drop shadow: shifted dilated mask, darken those pixels.
    if random.random() < p_shadow:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        sh_mask = cv2.dilate(mask, kernel, iterations=1)
        sx = random.randint(1, 2)
        sy = random.randint(1, 2)
        shifted = np.zeros_like(sh_mask)
        shifted[sy:, sx:] = sh_mask[: h - sy, : w - sx]
        sel = shifted > 0
        out[sel] = (out[sel].astype(np.int32) * 0.4).clip(0, 255).astype(np.uint8)

    # Black outline ring (dilated mask minus original mask).
    if random.random() < p_outline:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        ol_mask = cv2.dilate(mask, kernel, iterations=1)
        ring = cv2.subtract(ol_mask, mask)
        out[ring > 0] = [0, 0, 0]

    # White (or pale-yellow) text fill.
    if random.random() < 0.85:
        text_color = [255, 255, 255]
    else:
        text_color = [255, 240, 180]
    out[mask > 0] = text_color

    return out


# ---------------------------------------------------------------------------
# Copy-paste for rare-class oversampling
# ---------------------------------------------------------------------------


@dataclass
class CharCrop:
    image: np.ndarray  # small patch containing the rare character
    mask: np.ndarray | None  # alpha mask (optional, for blending)
    text: str  # the character/word


def copy_paste(
    base: Image,
    crops: list[CharCrop],
    max_paste: int = 3,
) -> tuple[Image, list[tuple[np.ndarray, str]]]:
    """Paste up to `max_paste` rare-character crops into the base image.

    Returns the augmented image and a list of (polygon, text) tuples to add
    to the detection/recognition labels.
    """
    out = base.copy()
    added: list[tuple[np.ndarray, str]] = []
    H, W = base.shape[:2]
    for _ in range(random.randint(1, max_paste)):
        if not crops:
            break
        crop = random.choice(crops)
        ch, cw = crop.image.shape[:2]
        if ch >= H or cw >= W:
            continue
        x = random.randint(0, W - cw)
        y = random.randint(0, H - ch)
        if crop.mask is not None:
            alpha = crop.mask[..., None] / 255.0
            out[y : y + ch, x : x + cw] = (
                out[y : y + ch, x : x + cw] * (1 - alpha) + crop.image * alpha
            ).astype(np.uint8)
        else:
            out[y : y + ch, x : x + cw] = crop.image
        poly = np.array(
            [[x, y], [x + cw, y], [x + cw, y + ch], [x, y + ch]], dtype=np.float32
        )
        added.append((poly, crop.text))
    return out, added


# ---------------------------------------------------------------------------
# Compose helper
# ---------------------------------------------------------------------------


@dataclass
class AugProb:
    fn: Callable[[Image], Image]
    p: float


class OCRAugment:
    """Probability-gated chain. Intended for recognition crops; detection uses
    its own polygon-aware pipeline elsewhere."""

    def __init__(self, ops: Iterable[AugProb]) -> None:
        self.ops = list(ops)

    def __call__(self, img: Image) -> Image:
        for op in self.ops:
            if random.random() < op.p:
                img = op.fn(img)
        return img


def default_recognition_augment() -> OCRAugment:
    return OCRAugment(
        [
            AugProb(lambda x: clahe(x), 0.3),
            AugProb(lambda x: speckle(x), 0.2),
            AugProb(lambda x: motion_blur(x), 0.2),
            AugProb(lambda x: jpeg_compress(x), 0.3),
            AugProb(lambda x: random_rotate(x, 3), 0.3),
            AugProb(lambda x: random_perspective(x, 0.02), 0.3),
        ]
    )


def dilated_crop_recognition_augment() -> OCRAugment:
    """iter 118 preset: warm-start rec to handle iter 113 dilated-crop dist.

    Combines `dilate_pad` (high prob 0.6 — primary signal) with default
    photometric ops (lower probs to keep training distribution close to the
    iter 89 v022 baseline). Geometric ops disabled (rec has plenty of
    geometric variation from `random_perspective` in default; we want this
    iter to focus the gradient on padding invariance).
    """
    return OCRAugment(
        [
            AugProb(lambda x: dilate_pad(x, max_h_ratio=0.25, max_w_ratio=0.06), 0.6),
            AugProb(lambda x: clahe(x), 0.2),
            AugProb(lambda x: speckle(x), 0.15),
            AugProb(lambda x: jpeg_compress(x), 0.2),
        ]
    )


def dilated_crop_aggressive_recognition_augment() -> OCRAugment:
    """iter 119 preset: more aggressive padding (max_h_ratio 0.35) to push
    dilation invariance further. Used with longer 4000-iter warm-start tail
    on top of v029 (iter 118 dilate_pad ckpt).

    p_fail ~50%: aggressive padding may overfit (rec predicts blanks where
    real test crops have only iter-118-level dilation), or main fixture beam
    path may regress in latency. Held1 -29.8% iter 118 may not extend.
    """
    return OCRAugment(
        [
            AugProb(lambda x: dilate_pad(x, max_h_ratio=0.35, max_w_ratio=0.10), 0.6),
            AugProb(lambda x: clahe(x), 0.2),
            AugProb(lambda x: speckle(x), 0.15),
            AugProb(lambda x: jpeg_compress(x), 0.2),
        ]
    )


def long_text_recognition_augment() -> OCRAugment:
    """iter 121 preset: aspect-ratio-aware aug routing.

    iter 120 audit localized residual mc to gt_len 16-20 bucket: 61-74% of
    edits across 3 fixtures, per-char error rate 4-8× short bucket. Short
    bucket is solved (94-96% exact-match). To avoid breaking solved short
    bucket, this preset routes heavier aug ONLY to long crops (AR ≥ 5,
    roughly ≥ 13 Korean chars at 48-h training input).

    Long-crop gating:
      - dilate_pad with max_w_ratio 0.18 (vs short fallback 0.06): tests
        horizontal-padding invariance specifically, where long crops have
        more room for pyclipper offset jitter.
      - motion_blur kernel 3-5 (lighter than default 3-9): per-char blur
        accumulates across long sequences; lighter kernel avoids
        catastrophic loss on short-text decoder path.
      - partial_occlusion 1-2 patches: simulates real-world per-char
        smudge / occlusion that's more likely on long captures.

    Short crops fall through with iter-118 baseline-like dilate_pad
    (max_w_ratio 0.06, no blur, no occlusion).

    p_fail ~50%: AR-conditional aug routing inside the dataloader is new
    code; the long-crop training distribution may not transfer well to real
    det-output crops (synth shape vs real DBNet output shape mismatch),
    or per-char blur could push the rec ckpt to predict shorter sequences
    (mode collapse on long bucket).
    """
    is_long = lambda x: x.shape[1] / max(x.shape[0], 1) >= 5.0
    return OCRAugment(
        [
            AugProb(
                lambda x: dilate_pad(x, max_h_ratio=0.25, max_w_ratio=0.18)
                if is_long(x)
                else dilate_pad(x, max_h_ratio=0.25, max_w_ratio=0.06),
                0.6,
            ),
            AugProb(lambda x: motion_blur(x, kernel_range=(3, 5)) if is_long(x) else x, 0.25),
            AugProb(lambda x: partial_occlusion(x) if is_long(x) else x, 0.20),
            AugProb(lambda x: clahe(x), 0.2),
            AugProb(lambda x: speckle(x), 0.15),
            AugProb(lambda x: jpeg_compress(x), 0.2),
        ]
    )


def real_photo_recognition_augment() -> OCRAugment:
    """iter 146 preset: real-photo distribution-shift attack.

    iter 143 honest bench (n=171 real Korean dialogue photos held-out):
        v031 None CER 47.80%
        v031 accurate CER 26.11%
        EasyOCR-only CER 22.87%
    v031's high-conf wrong predictions on real photos point to a
    training-time distribution mismatch — synthetic + EasyOCR-cleaned
    real_korean_pair labels do not expose the model to phone-screenshot
    JPEG artifacts, color casts, contrast issues, and motion-blur
    patterns common in dialogue captures.

    This preset stacks real-photo-mimic photometric ops on top of the
    iter 121 long_text mechanics (preserved so we don't regress the
    long-bucket gains v031 already has). Specifically:

      - HEAVIER jpeg_compress (q range 20-70 vs default 30-90) — phone
        screenshots are typically heavily-compressed.
      - HEAVIER speckle (sigma 0.12 vs default 0.08) — photo noise.
      - More frequent motion_blur (prob 0.35 vs default 0.20) — dialogue
        capture motion.
      - clahe at higher prob 0.40 — contrast normalization mimic.
      - Geometric ops: small rotate + perspective (already in default).

    p_fail ≥ 50%: heavier photometric aug may push the model to over-
    correct contrast / over-blur, regressing solved synthetic buckets.
    Or the underlying limit is label-noise (v031 trained on 22.87%-wrong
    EasyOCR labels), in which case aug alone won't move CER and we'd
    need to switch to higher-quality labels (out of scope for iter 146).
    """
    is_long = lambda x: x.shape[1] / max(x.shape[0], 1) >= 5.0
    return OCRAugment(
        [
            # iter 121 long_text mechanic preserved (AR-conditional dilate_pad)
            AugProb(
                lambda x: dilate_pad(x, max_h_ratio=0.25, max_w_ratio=0.18)
                if is_long(x)
                else dilate_pad(x, max_h_ratio=0.25, max_w_ratio=0.06),
                0.6,
            ),
            # Real-photo photometric — heavier than default and long_text presets
            AugProb(lambda x: jpeg_compress(x, quality_range=(20, 70)), 0.50),
            AugProb(lambda x: speckle(x, sigma=0.12), 0.30),
            AugProb(lambda x: motion_blur(x, kernel_range=(3, 7)), 0.35),
            AugProb(lambda x: clahe(x), 0.40),
            # Geometric — gentle to avoid blowing up warm-start loss
            AugProb(lambda x: random_rotate(x, 2), 0.20),
            AugProb(lambda x: random_perspective(x, 0.015), 0.20),
            # Long-only: partial occlusion (preserves iter 121 mechanic)
            AugProb(lambda x: partial_occlusion(x) if is_long(x) else x, 0.15),
        ]
    )


def subtitle_real_photo_recognition_augment() -> OCRAugment:
    """iter 162 preset: subtitle-style + real_photo combined.

    iter 161 confirmed dialogue raw 64% v039 vs 22% EasyOCR is REAL
    distribution mismatch on Korean drama subtitle visual style, not
    fixture noise. iter 146 real_photo aug helps phone-screenshot photo
    artifacts but does NOT introduce subtitle-style imagery (white-on-
    vibrant, outline stroke, drop shadow).

    This preset stacks `subtitle_overlay` at low prob 0.12 on top of
    iter 146 real_photo mechanics. 12% means ~1 in 8 crops is rendered
    in subtitle-style; rest stay synthetic-distribution to preserve the
    iter 159 v039 book WIN.

    p_fail ~50%: subtitle overlay synthetic-vs-real mismatch may not
    transfer (CER unchanged on dialogue, possibly small book regression
    from polarity-inversion exposure).
    """
    is_long = lambda x: x.shape[1] / max(x.shape[0], 1) >= 5.0
    return OCRAugment(
        [
            # iter 121 long_text mechanic preserved (AR-conditional dilate_pad)
            AugProb(
                lambda x: dilate_pad(x, max_h_ratio=0.25, max_w_ratio=0.18)
                if is_long(x)
                else dilate_pad(x, max_h_ratio=0.25, max_w_ratio=0.06),
                0.6,
            ),
            # NEW iter 162: subtitle-style overlay (~12% of crops)
            AugProb(lambda x: subtitle_overlay(x), 0.12),
            # iter 146 real_photo photometric (slightly down-weighted to
            # leave room for subtitle aug without overall aug pressure
            # ballooning).
            AugProb(lambda x: jpeg_compress(x, quality_range=(20, 70)), 0.45),
            AugProb(lambda x: speckle(x, sigma=0.12), 0.25),
            AugProb(lambda x: motion_blur(x, kernel_range=(3, 7)), 0.30),
            AugProb(lambda x: clahe(x), 0.35),
            AugProb(lambda x: random_rotate(x, 2), 0.20),
            AugProb(lambda x: random_perspective(x, 0.015), 0.20),
            AugProb(lambda x: partial_occlusion(x) if is_long(x) else x, 0.15),
        ]
    )


def decorative_recognition_augment() -> OCRAugment:
    """Geometry-light augment for cursive/decorative glyphs.

    _76 (iter 20): iter 18's new-data approach failed on decorative CER.
    Second-attempt hypothesis: existing `RecognitionTrainingDataset` applies
    NO augmentation at all — the model never sees stroke-width variation,
    slight rotation, or affine jitter that decorative fonts like
    NanumBrushScript, HiMelody, Gaegu exhibit naturally. This preset
    introduces those variations so weights learn glyph invariance.

    First try was too aggressive (stroke_jitter 0.5, rotate 6°) — warm-start
    loss blew up 0.5 → 375. Gentler preset: low probabilities, 1-iter
    stroke jitter only, tight 2° rotation. Warm-start-safe.
    """
    return OCRAugment(
        [
            AugProb(lambda x: stroke_width_jitter(x, max_iter=1), 0.15),
            AugProb(lambda x: random_rotate(x, 2), 0.2),
            AugProb(lambda x: random_perspective(x, 0.015), 0.15),
        ]
    )
