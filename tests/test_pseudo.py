import numpy as np

from spinaiocr.data.pseudo import ConsensusConfig, consensus, polygon_iou
from spinaiocr.teachers.base import TeacherLine, TeacherPrediction


def _box(x, y, w, h):
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)


def test_polygon_iou_identical():
    a = _box(0, 0, 10, 10)
    assert polygon_iou(a, a) == 1.0


def test_polygon_iou_disjoint():
    a = _box(0, 0, 10, 10)
    b = _box(100, 100, 10, 10)
    assert polygon_iou(a, b) == 0.0


def test_consensus_three_teachers_all_agree():
    bbox = _box(10, 10, 50, 20)
    preds = [
        TeacherPrediction(teacher=f"t{i}", lines=[TeacherLine(text="hello", bbox=bbox)])
        for i in range(3)
    ]
    words = consensus(preds, ConsensusConfig(iou_thresh=0.5, cer_thresh=0.1))
    assert len(words) == 1
    assert words[0].text == "hello"
    assert words[0].tier == "high"
    assert words[0].confidence == 1.0


def test_consensus_rejects_minority():
    bbox = _box(10, 10, 50, 20)
    preds = [
        TeacherPrediction(teacher="t1", lines=[TeacherLine(text="hello", bbox=bbox)]),
        TeacherPrediction(teacher="t2", lines=[TeacherLine(text="hello", bbox=bbox)]),
        TeacherPrediction(teacher="t3", lines=[TeacherLine(text="WRONG", bbox=bbox)]),
    ]
    words = consensus(preds, ConsensusConfig(iou_thresh=0.5, cer_thresh=0.1))
    assert len(words) == 1
    assert words[0].text == "hello"
    assert words[0].tier == "mid"
