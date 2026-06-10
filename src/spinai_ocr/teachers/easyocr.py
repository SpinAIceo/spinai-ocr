"""EasyOCR teacher."""
from __future__ import annotations

import time
import cv2
import numpy as np

from spinai_ocr.log import get_logger
from spinai_ocr.teachers.base import (
    ImageLike,
    OCRTeacher,
    TeacherLine,
    TeacherPrediction,
    _load_image,
    register,
)

log = get_logger("spinai_ocr.teachers.easyocr")


@register
class EasyOCRTeacher(OCRTeacher):
    name = "easyocr"
    license = "apache-2.0"
    commercial_ok = True

    def __init__(self, lang: str = "ko") -> None:
        super().__init__(lang=lang)
        try:
            import easyocr  # type: ignore
        except ImportError as e:
            log.error("teacher.easyocr.unavailable lang=%s err=%s", lang, e,
                      extra={"teacher": "easyocr", "lang": lang, "error": str(e)})
            raise ImportError("easyocr not installed. pip install easyocr") from e
        langs = [lang] if lang == "en" else [lang, "en"]
        t0 = time.perf_counter()
        self._engine = easyocr.Reader(langs, verbose=False)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        log.info("teacher.easyocr.loaded lang=%s langs=%s elapsed_ms=%.1f",
                 lang, langs, elapsed_ms,
                 extra={"teacher": "easyocr", "lang": lang, "langs": langs,
                        "elapsed_ms": round(elapsed_ms, 1)})

    def __call__(self, image: ImageLike) -> TeacherPrediction:
        arr = _load_image(image)
        # iter 134: _load_image returns RGB (PIL convention). EasyOCR's
        # reformat_input treats a 3-channel ndarray as BGR (see
        # `pipeline._easyocr_fallback` for full rationale). Pre-convert.
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        h, w = arr.shape[:2]
        t0 = time.perf_counter()
        raw = self._engine.readtext(arr, detail=1)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        lines = [
            TeacherLine(
                text=text,
                bbox=np.array(bbox, dtype=np.float32),
                confidence=float(conf),
            )
            for bbox, text, conf in raw
        ]
        log.debug("teacher.easyocr.infer lang=%s img=%dx%d lines=%d elapsed_ms=%.1f",
                  self.lang, w, h, len(lines), elapsed_ms,
                  extra={"teacher": "easyocr", "lang": self.lang,
                         "img_w": w, "img_h": h, "n_lines": len(lines),
                         "elapsed_ms": round(elapsed_ms, 1)})
        return TeacherPrediction(
            teacher=self.name, lines=lines, lang=self.lang, license=self.license
        )
