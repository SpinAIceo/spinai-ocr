"""PaddleOCR teacher."""
from __future__ import annotations

import numpy as np

from spinaiocr.teachers.base import (
    ImageLike,
    OCRTeacher,
    TeacherLine,
    TeacherPrediction,
    _load_image,
    register,
)


@register
class PaddleTeacher(OCRTeacher):
    name = "paddle"
    license = "apache-2.0"
    commercial_ok = True

    def __init__(self, lang: str = "ko") -> None:
        super().__init__(lang=lang)
        # PaddleOCR language codes differ from ISO 639-1
        paddle_lang = {"ko": "korean", "en": "en", "ja": "japan", "zh": "ch"}.get(lang, lang)
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except ImportError as e:
            raise ImportError(
                "paddleocr not installed. pip install paddleocr paddlepaddle"
            ) from e
        self._engine = PaddleOCR(lang=paddle_lang, use_angle_cls=True, show_log=False)

    def __call__(self, image: ImageLike) -> TeacherPrediction:
        arr = _load_image(image)
        raw = self._engine.ocr(arr, cls=True)
        lines: list[TeacherLine] = []
        for page in raw or []:
            for bbox, (text, conf) in page or []:
                lines.append(
                    TeacherLine(
                        text=text,
                        bbox=np.array(bbox, dtype=np.float32),
                        confidence=float(conf),
                    )
                )
        return TeacherPrediction(
            teacher=self.name, lines=lines, lang=self.lang, license=self.license
        )
