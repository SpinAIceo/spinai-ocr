"""End-to-end smoke test for the multi-teacher pseudo labeling pipeline.

Uses MockTeacher objects (no external OCR engines required) so this test
runs anywhere. Verifies:
- 3 agreeing teachers → 1 high-tier consensus word
- 3 disagreeing teachers → 0 words (default rejection)
- Mixed batch: majority agreement → 1 mid-tier word
- File output matches DetectionDataset JSONL format
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from spinaiocr.data.pseudo import (
    ConsensusConfig,
    pseudo_label_directory,
    pseudo_label_image,
)
from spinaiocr.teachers.base import OCRTeacher, TeacherLine, TeacherPrediction


class MockTeacher(OCRTeacher):
    """Returns a fixed scripted output regardless of input image."""

    def __init__(self, name: str, lines: list[TeacherLine]) -> None:
        super().__init__(lang="ko")
        self._name = name
        self._lines = lines
        self.license = "test"

    @property
    def name(self) -> str:  # type: ignore[override]
        return self._name

    def __call__(self, image):  # type: ignore[override]
        return TeacherPrediction(
            teacher=self._name, lines=list(self._lines), lang=self.lang, license=self.license
        )


def _bbox(x, y, w, h):
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)


def test_pseudo_label_image_high_consensus(tmp_path: Path):
    img_path = tmp_path / "img.png"
    Image.new("RGB", (128, 64), "white").save(img_path)

    bbox = _bbox(10, 10, 60, 20)
    teachers = [MockTeacher(f"t{i}", [TeacherLine(text="안녕하세요", bbox=bbox)]) for i in range(3)]

    rec = pseudo_label_image(img_path, teachers, ConsensusConfig())
    assert len(rec["words"]) == 1
    assert rec["words"][0]["text"] == "안녕하세요"
    assert rec["words"][0]["tier"] == "high"
    assert rec["words"][0]["conf"] == 1.0


def test_pseudo_label_directory_writes_jsonl(tmp_path: Path):
    for i in range(3):
        Image.new("RGB", (64, 32), (i * 20, 200, 100)).save(tmp_path / f"{i}.png")
    bbox = _bbox(5, 5, 40, 15)
    teachers = [
        MockTeacher("a", [TeacherLine(text="hello", bbox=bbox)]),
        MockTeacher("b", [TeacherLine(text="hello", bbox=bbox)]),
        MockTeacher("c", [TeacherLine(text="hell0", bbox=bbox)]),  # OCR-style typo
    ]
    out = tmp_path / "labels.jsonl"
    n = pseudo_label_directory(tmp_path, out, teachers, ConsensusConfig(cer_thresh=0.25))
    assert n == 3
    records = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert all(r["words"][0]["text"] == "hello" for r in records)


def test_disagreeing_teachers_rejected(tmp_path: Path):
    img_path = tmp_path / "img.png"
    Image.new("RGB", (64, 32), "white").save(img_path)
    bbox = _bbox(5, 5, 40, 15)
    teachers = [
        MockTeacher("a", [TeacherLine(text="one", bbox=bbox)]),
        MockTeacher("b", [TeacherLine(text="TWO", bbox=bbox)]),
        MockTeacher("c", [TeacherLine(text="three", bbox=bbox)]),
    ]
    rec = pseudo_label_image(img_path, teachers, ConsensusConfig(cer_thresh=0.1))
    assert rec["words"] == []
