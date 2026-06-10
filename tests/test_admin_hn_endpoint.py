"""iter 83: tests for /admin/hard_negatives auth + retrieve endpoints.

Auth model: SPINAI_ADMIN_TOKEN env var must be set AND request must send a
matching X-Admin-Token header. Mismatch (or env unset) returns 404 — same
as a missing endpoint, so unauthorized callers cannot detect existence.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image


def _make_queue(tmp_path: Path, uids: list[str]) -> Path:
    """Build a fake queue.jsonl + image dir, returns root path."""
    root = tmp_path / "hn"
    (root / "images").mkdir(parents=True, exist_ok=True)
    queue_lines = []
    for u in uids:
        img = Image.new("RGB", (32, 32), color=(127, 127, 127))
        img.save(root / "images" / f"{u}.jpg", "JPEG")
        queue_lines.append(json.dumps({
            "uid": u, "ts": "2026-04-26T00:00:00Z",
            "image_path": f"{root.as_posix()}/images/{u}.jpg",
            "mean_confidence": 0.42,
            "threshold": 0.60,
            "n_lines": 1,
            "lines": [{"text": "test", "confidence": 0.42, "bbox": []}],
            "request_meta": {"lang": "ko", "tier": "consumer_v1"},
        }))
    (root / "queue.jsonl").write_text("\n".join(queue_lines) + "\n", encoding="utf-8")
    return root


@pytest.fixture
def client(tmp_path, monkeypatch):
    root = _make_queue(tmp_path, ["abc123", "def456"])
    monkeypatch.setenv("SPINAI_HN_DIR", str(root))
    from spinai_ocr.serve.app import app
    return TestClient(app), root


def test_list_no_token_env_returns_404(client, monkeypatch):
    c, _ = client
    monkeypatch.delenv("SPINAI_ADMIN_TOKEN", raising=False)
    r = c.get("/admin/hard_negatives/list")
    assert r.status_code == 404


def test_list_token_env_set_but_no_header_returns_404(client, monkeypatch):
    c, _ = client
    monkeypatch.setenv("SPINAI_ADMIN_TOKEN", "secret_xyz")
    r = c.get("/admin/hard_negatives/list")
    assert r.status_code == 404


def test_list_token_mismatch_returns_404(client, monkeypatch):
    c, _ = client
    monkeypatch.setenv("SPINAI_ADMIN_TOKEN", "secret_xyz")
    r = c.get("/admin/hard_negatives/list",
               headers={"X-Admin-Token": "wrong"})
    assert r.status_code == 404


def test_list_correct_token_returns_entries(client, monkeypatch):
    c, _ = client
    monkeypatch.setenv("SPINAI_ADMIN_TOKEN", "secret_xyz")
    r = c.get("/admin/hard_negatives/list",
               headers={"X-Admin-Token": "secret_xyz"})
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == 2
    assert body["n_images_on_disk"] == 2
    assert body["total_disk_mb"] >= 0
    uids = {e["uid"] for e in body["entries"]}
    assert uids == {"abc123", "def456"}


def test_list_empty_queue_returns_n0(tmp_path, monkeypatch):
    root = tmp_path / "empty_hn"
    root.mkdir()
    monkeypatch.setenv("SPINAI_HN_DIR", str(root))
    monkeypatch.setenv("SPINAI_ADMIN_TOKEN", "tok")
    from spinai_ocr.serve.app import app
    c = TestClient(app)
    r = c.get("/admin/hard_negatives/list", headers={"X-Admin-Token": "tok"})
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == 0
    assert body["entries"] == []


def test_image_get_correct_token_returns_jpeg(client, monkeypatch):
    c, _ = client
    monkeypatch.setenv("SPINAI_ADMIN_TOKEN", "secret_xyz")
    r = c.get("/admin/hard_negatives/image/abc123",
               headers={"X-Admin-Token": "secret_xyz"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    img = Image.open(io.BytesIO(r.content))
    assert img.size == (32, 32)


def test_image_get_unknown_uid_returns_404(client, monkeypatch):
    c, _ = client
    monkeypatch.setenv("SPINAI_ADMIN_TOKEN", "secret_xyz")
    r = c.get("/admin/hard_negatives/image/zzz999",
               headers={"X-Admin-Token": "secret_xyz"})
    assert r.status_code == 404


def test_image_get_traversal_blocked(client, monkeypatch):
    """Non-alphanumeric uid (path traversal attempt) returns 400."""
    c, _ = client
    monkeypatch.setenv("SPINAI_ADMIN_TOKEN", "secret_xyz")
    r = c.get("/admin/hard_negatives/image/..%2Fpasswd",
               headers={"X-Admin-Token": "secret_xyz"})
    assert r.status_code in (400, 404)


def test_image_get_no_token_returns_404(client, monkeypatch):
    c, _ = client
    monkeypatch.delenv("SPINAI_ADMIN_TOKEN", raising=False)
    r = c.get("/admin/hard_negatives/image/abc123")
    assert r.status_code == 404
