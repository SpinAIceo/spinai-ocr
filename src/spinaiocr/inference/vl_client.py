# -*- coding: utf-8 -*-
"""Client for the optional PaddleOCR-VL accuracy-tier service (scripts/vl_service.py).

The product's `vlm` routing mode recognizes detected crops via this service. On
ANY failure (service down/unreachable/error) calls return the caller's fallback,
so the `vlm` path can never degrade below the EasyOCR/edge text it replaces.

Endpoint is configurable via SPINAI_VL_URL (default http://localhost:8800).
"""
from __future__ import annotations
import base64
import io
import json
import os
import urllib.request

VL_URL = os.environ.get("SPINAI_VL_URL", "http://localhost:8800").rstrip("/")


def vl_healthy(timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(f"{VL_URL}/health", timeout=timeout) as r:
            return json.loads(r.read()).get("status") == "ok"
    except Exception:
        return False


def vl_ocr(pil_image, fallback: str = "", max_new_tokens: int = 48, timeout: float = 30.0) -> str:
    """Recognize text in a (cropped) PIL image via the VL service.
    Returns `fallback` on any error (graceful degradation)."""
    try:
        buf = io.BytesIO()
        pil_image.convert("RGB").save(buf, format="JPEG")
        body = json.dumps({
            "image_b64": base64.b64encode(buf.getvalue()).decode(),
            "max_new_tokens": max_new_tokens,
        }).encode()
        req = urllib.request.Request(
            f"{VL_URL}/ocr", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read()).get("text", fallback)
    except Exception:
        return fallback
