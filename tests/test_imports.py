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
    "spinaiocr",
    "spinaiocr.cli",
    "spinaiocr.config",
    # Models
    "spinaiocr.models.detection",
    "spinaiocr.models.recognition",
    "spinaiocr.models.angle_cls",
    "spinaiocr.models.dexined",
    "spinaiocr.models.backbones",
    "spinaiocr.models.registry",
    "spinaiocr.models.weights",
    # Inference
    "spinaiocr.inference.pipeline",
    "spinaiocr.inference.db_postprocess",
    "spinaiocr.inference.decoders",
    "spinaiocr.inference.tta_ensemble",
    "spinaiocr.inference.edge_refine",
    "spinaiocr.inference.pdf_fallback",
    "spinaiocr.inference.longwidth",
    # Data
    "spinaiocr.data.dataset",
    "spinaiocr.data.synth_simple",
    "spinaiocr.data.synth_detection",
    "spinaiocr.data.augment",
    "spinaiocr.data.detection_augment",
    "spinaiocr.data.collect",
    "spinaiocr.data.adapters",
    "spinaiocr.data.sources",
    "spinaiocr.data.validate",
    "spinaiocr.data.pseudo",
    "spinaiocr.data.mining",
    "spinaiocr.data.active_learning",
    # Training
    "spinaiocr.training.datamodule",
    "spinaiocr.training.losses",
    "spinaiocr.training.db_gt",
    "spinaiocr.training.mean_teacher",
    "spinaiocr.training.uncertainty",
    "spinaiocr.training.distill",
    "spinaiocr.training.curriculum",
    "spinaiocr.training.continuous",
    # Benchmark
    "spinaiocr.benchmark.metrics",
    "spinaiocr.benchmark.eval_harness",
    "spinaiocr.benchmark.error_analysis",
    "spinaiocr.benchmark.run",
    # Teachers
    "spinaiocr.teachers.base",
    "spinaiocr.teachers.paddle",
    "spinaiocr.teachers.easyocr",
    "spinaiocr.teachers.tesseract",
    "spinaiocr.teachers.trocr",
    "spinaiocr.teachers.gemma",
    # Layout
    "spinaiocr.layout.analyzer",
    "spinaiocr.layout.table",
    "spinaiocr.layout.formula",
    # Plumbing
    "spinaiocr.experiments.tracker",
    "spinaiocr.deploy.git_ops",
    "spinaiocr.deploy.workflows",
    "spinaiocr.deploy.onnx_export",
    "spinaiocr.io",
    "spinaiocr.log",
    "spinaiocr.serve.app",
    "spinaiocr.vocab.base",
    "spinaiocr.postprocess.llm",
]


@pytest.mark.parametrize("module", MODULES)
def test_import_clean(module: str) -> None:
    importlib.import_module(module)
