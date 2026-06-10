import cv2
import numpy as np

from spinai_ocr.inference.db_postprocess import DBPostProcessor


def test_extract_polygon_from_synthetic_blob():
    prob = np.zeros((128, 256), dtype=np.float32)
    cv2.rectangle(prob, (40, 40), (200, 80), 0.9, -1)
    pp = DBPostProcessor(thresh=0.3, box_thresh=0.5, unclip_ratio=1.0, min_box_size=5)
    polys = pp.extract_polygons(prob, (128, 256))
    assert len(polys) >= 1
    poly = polys[0]
    # with unclip_ratio=1.0 the box expands by ~area/length ≈ 16 px; expect the
    # polygon to cover the blob + a margin, and to stay inside the image
    assert poly[:, 0].min() < 40  # expanded leftward
    assert poly[:, 0].max() > 200  # expanded rightward
    assert poly[:, 0].max() <= 255
    assert poly[:, 1].max() <= 127


def test_low_prob_rejected():
    prob = np.full((64, 64), 0.1, dtype=np.float32)
    pp = DBPostProcessor()
    assert pp.extract_polygons(prob, (64, 64)) == []
