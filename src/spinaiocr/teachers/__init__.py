"""Teacher OCR wrappers for multi-teacher pseudo labeling.

Each teacher implements :class:`OCRTeacher` and returns a list of
:class:`TeacherPrediction` objects. The consensus module in
``spinaiocr.data.pseudo`` merges them into pseudo labels.
"""
from spinaiocr.teachers.base import OCRTeacher, TeacherPrediction, build_teacher

__all__ = ["OCRTeacher", "TeacherPrediction", "build_teacher"]
