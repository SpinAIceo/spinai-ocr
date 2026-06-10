"""Benchmark harness — CER/WER/FPS + competitor comparison."""
from spinaiocr.benchmark.metrics import compute_cer, compute_wer, MetricResult

__all__ = ["compute_cer", "compute_wer", "MetricResult"]
