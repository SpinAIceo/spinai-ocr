"""Quota is charged only for successful /ocr calls, not malformed ones.

Pinned in _55 after /loop live probe. Pre-fix, every /ocr attempt bumped
the quota counter — an oversized-image 413 burned the caller's quota
through no fault of their own. Fix: check_monthly_quota() gates without
incrementing; commit_monthly_call() runs after the response body is
ready.
"""
from __future__ import annotations

import io

from fastapi.testclient import TestClient
from PIL import Image

from spinaiocr.serve.app import app
from spinaiocr.serve.ratelimit import get_usage, reset_all, set_quota_override


def _png_bytes(w: int = 256, h: int = 64) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "white").save(buf, format="PNG")
    return buf.getvalue()


def test_successful_ocr_commits_quota():
    reset_all()
    c = TestClient(app)
    r = c.post("/ocr", files={"file": ("t.png", _png_bytes(), "image/png")})
    assert r.status_code == 200, r.text
    # Identity is the ipv4 of the TestClient; inspect directly via get_usage
    used = sum(v for v in _all_counts().values())
    assert used == 1


def test_413_does_not_burn_quota():
    """Oversized upload must 413 without consuming quota."""
    reset_all()
    c = TestClient(app)
    from spinaiocr.serve.app import MAX_IMAGE_BYTES
    big = b"\xff" * (MAX_IMAGE_BYTES + 10)
    r = c.post("/ocr", files={"file": ("big.bin", big, "image/png")})
    assert r.status_code == 413
    assert sum(_all_counts().values()) == 0


def test_400_invalid_image_does_not_burn_quota():
    reset_all()
    c = TestClient(app)
    r = c.post("/ocr", files={"file": ("junk.png", b"not-an-image", "image/png")})
    assert r.status_code == 400
    assert sum(_all_counts().values()) == 0


def test_batch_quota_rejects_when_budget_too_small():
    """Batch of N images needs N headroom; over-budget should 429."""
    reset_all()
    c = TestClient(app)
    # Seed the counter to 49 so a batch of 2 would overflow limit=50.
    from spinaiocr.serve.ratelimit import _COUNTS, _month_key
    identity = "ip:testclient"
    _COUNTS[(identity, _month_key())] = 49
    import base64
    b64 = base64.b64encode(_png_bytes()).decode()
    r = c.post("/ocr/batch", json={"images": [b64, b64], "lang": "ko",
                                     "tier": "lite"})
    assert r.status_code == 429, r.text
    # Failed batch must not increment counter
    assert _COUNTS[(identity, _month_key())] == 49


def _all_counts() -> dict:
    from spinaiocr.serve.ratelimit import _COUNTS
    return dict(_COUNTS)


def test_422_high_unk_ratio_does_not_burn_quota(monkeypatch):
    """iter 135 pin: /ocr's 422 unk-ratio gate ≥ 0.20 must not burn quota
    (free retry per medical-safety design). Pre-iter134 lacked this pin —
    422 path could regress silently into quota commits."""
    reset_all()
    c = TestClient(app)
    # Force the pipeline to return a synthetic high-unk result without
    # depending on real OCR output (which is empty for blank canvas).
    from spinaiocr.inference.pipeline import OCRPipeline, OCRLine, OCRResult

    def _fake_call(self, image, single_line=None, decode_mode=None):  # noqa: ANN001
        return OCRResult(
            lines=[
                OCRLine(text="?? ?", bbox=[(0.0, 0.0)] * 4,
                        confidence=0.5, unk_count=3, min_char_conf=0.5),
            ],
            image_width=256, image_height=64,
        )

    monkeypatch.setattr(OCRPipeline, "__call__", _fake_call)
    r = c.post("/ocr", files={"file": ("t.png", _png_bytes(), "image/png")})
    assert r.status_code == 422, r.text
    assert sum(_all_counts().values()) == 0


def test_batch_uncertain_items_do_not_burn_quota(monkeypatch):
    """iter 135 fix: batch parity with /ocr — uncertain items (status
    'uncertain') do not consume quota. Mixed batch with k uncertain + m ok
    consumes only m, not k+m."""
    reset_all()
    c = TestClient(app)

    from spinaiocr.inference.pipeline import OCRPipeline, OCRLine, OCRResult

    # iter 155: route by image content, not call order. /ocr/batch fans
    # out via asyncio.gather so the order in which _fake_call observes
    # the two requests is non-deterministic — keying on a hashable
    # property of `image` keeps the test order-independent.
    def _fake_call(self, image, single_line=None, decode_mode=None):  # noqa: ANN001
        # The two mock images differ only by width (256 vs 257) so the
        # caller can identify which case to return without depending on
        # invocation order.
        if image.size[0] == 256:
            return OCRResult(
                lines=[OCRLine(text="안녕", bbox=[(0.0, 0.0)] * 4,
                               confidence=0.9, unk_count=0, min_char_conf=0.9)],
                image_width=256, image_height=64,
            )
        return OCRResult(
            lines=[OCRLine(text="???", bbox=[(0.0, 0.0)] * 4,
                           confidence=0.5, unk_count=3, min_char_conf=0.5)],
            image_width=257, image_height=64,
        )

    monkeypatch.setattr(OCRPipeline, "__call__", _fake_call)
    import base64
    ok_b64 = base64.b64encode(_png_bytes(w=256)).decode()
    unk_b64 = base64.b64encode(_png_bytes(w=257)).decode()
    r = c.post("/ocr/batch", json={"images": [ok_b64, unk_b64], "lang": "ko",
                                     "tier": "lite"})
    assert r.status_code == 200, r.text
    items = r.json()
    assert items[0]["status"] == "ok"
    assert items[1]["status"] == "uncertain"
    # Only the OK item burned quota.
    assert sum(_all_counts().values()) == 1
