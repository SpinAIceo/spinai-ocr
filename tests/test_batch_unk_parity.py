"""/ocr/batch should propagate <unk> stats and status per item.

Pre-fix, /ocr/batch stripped unk_count from each Line and ignored the
medical-safety gate — callers using batch lost the safety rail that
single /ocr enforces at 20% unk_ratio. This pins per-item parity:
unk_count on Line, status on OCRResponse, total_unk aggregate.
"""
from __future__ import annotations

import base64
import io

from fastapi.testclient import TestClient
from PIL import Image

from spinai_ocr.serve.app import app
from spinai_ocr.serve.ratelimit import reset_all


def _b64() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (256, 64), "white").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_batch_items_expose_status_and_unk_fields():
    reset_all()
    c = TestClient(app)
    r = c.post("/ocr/batch", json={"images": [_b64(), _b64()],
                                     "lang": "ko", "tier": "lite"})
    assert r.status_code == 200, r.text
    items = r.json()
    assert len(items) == 2
    for item in items:
        # Schema parity with /ocr response
        assert "status" in item
        assert item["status"] in {"ok", "partial", "uncertain"}
        assert "total_unk" in item
        assert isinstance(item["total_unk"], int)
        # And per-line unk_count propagated
        for line in item["lines"]:
            assert "unk_count" in line


def test_batch_uncertain_items_count_header_absent_on_clean_input():
    reset_all()
    c = TestClient(app)
    r = c.post("/ocr/batch", json={"images": [_b64()], "lang": "ko",
                                     "tier": "lite"})
    assert r.status_code == 200
    # Blank image → no unk → no X-Uncertain-Items header
    assert "x-uncertain-items" not in {k.lower() for k in r.headers.keys()}
