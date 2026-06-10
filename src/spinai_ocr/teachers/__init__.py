"""Teacher OCR wrappers for multi-teacher pseudo labeling.

Each teacher implements :class:`OCRTeacher` and returns a list of
:class:`TeacherPrediction` objects. The consensus module in
``spinai_ocr.data.pseudo`` merges them into pseudo labels.
"""
from spinai_ocr.teachers.base import OCRTeacher, TeacherPrediction, build_teacher

__all__ = ["OCRTeacher", "TeacherPrediction", "build_teacher"]
