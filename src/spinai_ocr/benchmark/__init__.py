"""Benchmark harness — CER/WER/FPS + competitor comparison."""
from spinai_ocr.benchmark.metrics import compute_cer, compute_wer, MetricResult

__all__ = ["compute_cer", "compute_wer", "MetricResult"]
