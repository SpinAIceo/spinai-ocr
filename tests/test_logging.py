"""Tests for the structured logging system."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest

from spinai_ocr.log import (
    NaNInfGuard,
    capture_crashes,
    check_finite,
    format_tensor_brief,
    get_logger,
    log_span,
    setup_logging,
)
from spinai_ocr.log.config import LogConfig


@pytest.fixture
def log_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SPINAI_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("SPINAI_LOG_LEVEL", "DEBUG")
    # reset global state
    import spinai_ocr.log.config as cfg_mod
    cfg_mod._CONFIGURED = False
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    setup_logging(force=True)
    yield tmp_path
    # cleanup handlers
    for h in list(root.handlers):
        root.removeHandler(h)
    cfg_mod._CONFIGURED = False


def test_setup_creates_log_dir(log_env):
    assert (log_env / "crashes").exists()
    # emit a record so the text file gets created
    get_logger("test").info("hello")
    for h in logging.getLogger().handlers:
        h.flush()
    assert (log_env / "spinai.log").exists()


def test_jsonl_records_contain_extra(log_env):
    log = get_logger("test.jsonl")
    log.warning("a_message", extra={"foo": 42, "bar": "baz"})
    for h in logging.getLogger().handlers:
        h.flush()
    jsonl = (log_env / "spinai.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in jsonl]
    last = records[-1]
    assert last["level"] == "WARNING"
    assert last["msg"] == "a_message"
    assert last["extra"]["foo"] == 42
    assert last["extra"]["bar"] == "baz"


def test_capture_crashes_writes_dump(log_env):
    with pytest.raises(ValueError):
        with capture_crashes("unit_test", extra={"step": 7, "shape": [3, 4]}):
            raise ValueError("boom")
    dumps = list((log_env / "crashes").glob("unit_test_*.json"))
    assert len(dumps) == 1
    payload = json.loads(dumps[0].read_text(encoding="utf-8"))
    assert payload["tag"] == "unit_test"
    assert payload["exception"]["type"] == "ValueError"
    assert payload["exception"]["message"] == "boom"
    assert payload["extra"]["step"] == 7
    assert "Traceback" in payload["exception"]["traceback"]


def test_capture_crashes_no_reraise(log_env):
    # should NOT raise when reraise=False
    with capture_crashes("no_reraise", reraise=False):
        raise RuntimeError("quiet")
    dumps = list((log_env / "crashes").glob("no_reraise_*.json"))
    assert len(dumps) == 1


def test_nan_guard_trips_after_max_bad(log_env):
    import torch
    guard = NaNInfGuard(max_bad=2)
    bad = torch.tensor([float("nan"), 1.0])
    with pytest.raises(FloatingPointError):
        guard.check("x", bad)
        guard.check("x", bad)


def test_check_finite_passes_clean(log_env):
    import torch
    assert check_finite("good", torch.tensor([1.0, 2.0]))
    assert check_finite("good_float", 0.5)


def test_check_finite_raises_on_nan(log_env):
    import torch
    with pytest.raises(FloatingPointError):
        check_finite("bad", torch.tensor([float("inf")]))


def test_format_tensor_brief(log_env):
    import torch
    s = format_tensor_brief(torch.zeros(2, 3))
    assert "shape=(2, 3)" in s
    assert "dtype=" in s


def test_log_span_emits_start_and_end(log_env):
    with log_span("a_span", iters=5):
        pass
    for h in logging.getLogger().handlers:
        h.flush()
    jsonl = (log_env / "spinai.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(l) for l in jsonl]
    span_events = [r for r in records if r.get("extra", {}).get("span") == "a_span"]
    kinds = {r["extra"]["event"] for r in span_events}
    assert kinds == {"start", "end"}


def test_per_module_override_via_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SPINAI_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("SPINAI_LOG_LEVEL", "WARNING")
    monkeypatch.setenv("SPINAI_LOG_LEVELS", '{"spinai_ocr.data": "DEBUG"}')
    import spinai_ocr.log.config as cfg_mod
    cfg_mod._CONFIGURED = False
    for h in list(logging.getLogger().handlers):
        logging.getLogger().removeHandler(h)
    setup_logging(force=True)
    # spinai_ocr.data should be DEBUG-level
    assert logging.getLogger("spinai_ocr.data").level == logging.DEBUG
    cfg_mod._CONFIGURED = False
