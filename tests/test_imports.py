"""All public modules must import cleanly.

Catches:
- ImportError from missing optional deps leaking into required path
- Circular imports
- Typos in new __init__.py files
- Modules that accidentally require GPU at import time
"""
from __future__ import annotations

import importlib

import pytest

MODULES = [
    # Core
    "spinai_ocr",
    "spinai_ocr.cli",
    "spinai_ocr.config",
    # Models
    "spinai_ocr.models.detection",
    "spinai_ocr.models.recognition",
    "spinai_ocr.models.angle_cls",
    "spinai_ocr.models.dexined",
    "spinai_ocr.models.backbones",
    "spinai_ocr.models.registry",
    "spinai_ocr.models.weights",
    # Inference
    "spinai_ocr.inference.pipeline",
    "spinai_ocr.inference.db_postprocess",
    "spinai_ocr.inference.decoders",
    "spinai_ocr.inference.tta_ensemble",
    "spinai_ocr.inference.edge_refine",
    "spinai_ocr.inference.pdf_fallback",
    "spinai_ocr.inference.longwidth",
    # Data
    "spinai_ocr.data.dataset",
    "spinai_ocr.data.synth_simple",
    "spinai_ocr.data.synth_detection",
    "spinai_ocr.data.augment",
    "spinai_ocr.data.detection_augment",
    "spinai_ocr.data.collect",
    "spinai_ocr.data.adapters",
    "spinai_ocr.data.sources",
    "spinai_ocr.data.validate",
    "spinai_ocr.data.pseudo",
    "spinai_ocr.data.mining",
    "spinai_ocr.data.active_learning",
    # Training
    "spinai_ocr.training.datamodule",
    "spinai_ocr.training.losses",
    "spinai_ocr.training.db_gt",
    "spinai_ocr.training.mean_teacher",
    "spinai_ocr.training.uncertainty",
    "spinai_ocr.training.distill",
    "spinai_ocr.training.curriculum",
    "spinai_ocr.training.continuous",
    # Benchmark
    "spinai_ocr.benchmark.metrics",
    "spinai_ocr.benchmark.eval_harness",
    "spinai_ocr.benchmark.error_analysis",
    "spinai_ocr.benchmark.run",
    # Teachers
    "spinai_ocr.teachers.base",
    "spinai_ocr.teachers.paddle",
    "spinai_ocr.teachers.easyocr",
    "spinai_ocr.teachers.tesseract",
    "spinai_ocr.teachers.trocr",
    "spinai_ocr.teachers.gemma",
    # Layout
    "spinai_ocr.layout.analyzer",
    "spinai_ocr.layout.table",
    "spinai_ocr.layout.formula",
    # Plumbing
    "spinai_ocr.experiments.tracker",
    "spinai_ocr.deploy.git_ops",
    "spinai_ocr.deploy.workflows",
    "spinai_ocr.deploy.onnx_export",
    "spinai_ocr.io",
    "spinai_ocr.log",
    "spinai_ocr.serve.app",
    "spinai_ocr.vocab.base",
    "spinai_ocr.postprocess.llm",
]


@pytest.mark.parametrize("module", MODULES)
def test_import_clean(module: str) -> None:
    importlib.import_module(module)
