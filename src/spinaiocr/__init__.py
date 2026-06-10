"""SPINAI OCR — Korean-first open-source OCR engine."""

__version__ = "0.0.1"

from spinaiocr.inference.pipeline import OCRPipeline, OCRLine, OCRResult

__all__ = ["OCRPipeline", "OCRLine", "OCRResult", "__version__"]
