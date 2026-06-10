"""TrOCR teacher (HuggingFace transformers).

TrOCR is recognition-only (no detection) — use it on cropped line images,
or pair it with a separate detector (e.g., our DBNet) for pseudo labeling of
the recognition head.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from spinaiocr.teachers.base import (
    ImageLike,
    OCRTeacher,
    TeacherLine,
    TeacherPrediction,
    _load_image,
    register,
)


@register
class TrOCRTeacher(OCRTeacher):
    name = "trocr"
    license = "mit"
    commercial_ok = True

    _DEFAULT_CKPT = {
        "en": "microsoft/trocr-base-printed",
        "handwritten": "microsoft/trocr-base-handwritten",
    }

    def __init__(
        self,
        lang: str = "en",
        checkpoint: str | None = None,
        device: str = "cpu",
    ) -> None:
        super().__init__(lang=lang)
        if checkpoint is None:
            checkpoint = self._DEFAULT_CKPT.get(lang, self._DEFAULT_CKPT["en"])
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel  # type: ignore
        except ImportError as e:
            raise ImportError(
                "transformers not installed. pip install transformers"
            ) from e
        self.processor = TrOCRProcessor.from_pretrained(checkpoint)
        self.model = VisionEncoderDecoderModel.from_pretrained(checkpoint)
        self.device = device
        self.model.to(device)
        self.model.eval()

    def __call__(self, image: ImageLike) -> TeacherPrediction:
        arr = _load_image(image)
        pil = Image.fromarray(arr)
        pixel_values = self.processor(pil, return_tensors="pt").pixel_values.to(self.device)
        import torch

        with torch.no_grad():
            ids = self.model.generate(pixel_values)
        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0]
        h, w = arr.shape[:2]
        bbox = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        line = TeacherLine(text=text, bbox=bbox, confidence=1.0)
        return TeacherPrediction(
            teacher=self.name, lines=[line], lang=self.lang, license=self.license
        )
