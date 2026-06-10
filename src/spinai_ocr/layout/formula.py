"""Math formula recognition.

Wraps `pix2tex` (LaTeX-OCR) and/or Nougat for inline/block equations.
Both are optional — if the package is missing, `recognize()` returns None
so the main pipeline can degrade gracefully.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image


Backend = Literal["pix2tex", "nougat"]


@dataclass
class FormulaResult:
    latex: str
    backend: str
    confidence: float = 1.0


class FormulaRecognizer:
    def __init__(self, backend: Backend = "pix2tex") -> None:
        self.backend = backend
        self._engine = None

    def _load(self) -> None:
        if self._engine is not None:
            return
        if self.backend == "pix2tex":
            try:
                from pix2tex.cli import LatexOCR  # type: ignore
            except ImportError as e:
                raise ImportError("pip install pix2tex") from e
            self._engine = LatexOCR()
        elif self.backend == "nougat":
            raise NotImplementedError("Nougat backend: wire transformers pipeline here.")
        else:
            raise ValueError(self.backend)

    def recognize(self, image: np.ndarray | Image.Image) -> FormulaResult | None:
        self._load()
        assert self._engine is not None
        pil = image if isinstance(image, Image.Image) else Image.fromarray(image)
        latex = self._engine(pil)
        return FormulaResult(latex=latex or "", backend=self.backend)
