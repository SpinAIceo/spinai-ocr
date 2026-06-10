"""Experiment tracking.

Records every training run in a structured form so we can compare CER/loss
across runs, reproduce, and gate regressions.

Quick start::

    from spinai_ocr.experiments import ExperimentTracker
    tracker = ExperimentTracker.create("reco_ko_subset", config=cfg_dict)
    tracker.log_metrics(step=100, cer=0.3, loss=1.2)
    tracker.log_artifact(ckpt_path, kind="checkpoint")
    tracker.finish(final_metrics={"best_cer": 0.1128})
"""
from spinai_ocr.experiments.tracker import ExperimentTracker, load_run

__all__ = ["ExperimentTracker", "load_run"]
