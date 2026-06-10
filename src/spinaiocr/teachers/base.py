"""Teacher OCR abstract interface.

All teachers return a common :class:`TeacherPrediction` schema so the
consensus module can vote across heterogeneous engines.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

from spinaiocr.log import get_logger

log = get_logger("spinaiocr.teachers")

ImageLike = Union[str, Path, np.ndarray, Image.Image]


@dataclass
class TeacherLine:
    text: str
    bbox: np.ndarray  # [4, 2] polygon (clockwise, top-left first)
    confidence: float = 1.0


@dataclass
class TeacherPrediction:
    teacher: str
    lines: list[TeacherLine] = field(default_factory=list)
    lang: str = "ko"
    license: str = "unknown"  # license of the teacher (gates downstream use)

    @property
    def full_text(self) -> str:
        return " ".join(l.text for l in self.lines)


class OCRTeacher(ABC):
    """Abstract base for any OCR engine acting as a teacher."""

    name: str = "base"
    license: str = "unknown"
    commercial_ok: bool = False

    def __init__(self, lang: str = "ko") -> None:
        self.lang = lang

    @abstractmethod
    def __call__(self, image: ImageLike) -> TeacherPrediction: ...


_REGISTRY: dict[str, type[OCRTeacher]] = {}


def register(cls: type[OCRTeacher]) -> type[OCRTeacher]:
    _REGISTRY[cls.name] = cls
    return cls


def build_teacher(name: str, **kwargs) -> OCRTeacher:
    if name not in _REGISTRY:
        # trigger module registration
        for mod in ("paddle", "easyocr", "tesseract", "trocr", "gemma"):
            try:
                __import__(f"spinaiocr.teachers.{mod}", fromlist=["*"])
            except ImportError as e:
                log.debug("teachers.%s unavailable: %s", mod, e,
                          extra={"teacher": mod, "error": str(e)})
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown teacher: {name}. Known: {sorted(_REGISTRY)}"
        )
    try:
        inst = _REGISTRY[name](**kwargs)
        log.info("teacher.built name=%s license=%s commercial_ok=%s",
                 name, inst.license, inst.commercial_ok,
                 extra={"teacher": name, "license": inst.license,
                        "commercial_ok": inst.commercial_ok})
        return inst
    except Exception as e:
        log.error("teacher.build_failed name=%s err=%s", name, e,
                  exc_info=True, extra={"teacher": name})
        raise


def _load_image(x: ImageLike) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, Image.Image):
        return np.asarray(x.convert("RGB"))
    return np.asarray(Image.open(x).convert("RGB"))
