"""Tesseract teacher (via pytesseract)."""
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
class TesseractTeacher(OCRTeacher):
    name = "tesseract"
    license = "apache-2.0"
    commercial_ok = True

    def __init__(self, lang: str = "ko") -> None:
        super().__init__(lang=lang)
        try:
            import pytesseract  # type: ignore
        except ImportError as e:
            raise ImportError(
                "pytesseract not installed. pip install pytesseract"
            ) from e
        self._pytess = pytesseract
        # Tesseract uses 3-letter codes
        self._tcode = {"ko": "kor", "en": "eng", "ja": "jpn", "zh": "chi_sim"}.get(
            lang, lang
        )

    def __call__(self, image: ImageLike) -> TeacherPrediction:
        arr = _load_image(image)
        data = self._pytess.image_to_data(
            arr, lang=self._tcode, output_type=self._pytess.Output.DICT
        )
        lines: list[TeacherLine] = []
        n = len(data["text"])
        for i in range(n):
            text = data["text"][i].strip()
            if not text:
                continue
            x, y, w, h = (
                int(data["left"][i]),
                int(data["top"][i]),
                int(data["width"][i]),
                int(data["height"][i]),
            )
            bbox = np.array(
                [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32
            )
            conf_raw = data["conf"][i]
            conf = float(conf_raw) / 100.0 if conf_raw not in ("-1", -1) else 0.0
            lines.append(TeacherLine(text=text, bbox=bbox, confidence=conf))
        return TeacherPrediction(
            teacher=self.name, lines=lines, lang=self.lang, license=self.license
        )
