"""Lightning training entry point.

Example:
    python -m spinai_ocr.training.train \
        --task recognition --config configs/recognition_ko_lite.yaml
"""
from __future__ import annotations

from pathlib import Path

import click
import yaml


@click.command()
@click.option("--task", type=click.Choice(["detection", "recognition"]), required=True)
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--checkpoints", default="checkpoints", type=click.Path())
@click.option("--wandb/--no-wandb", default=False)
def main(task: str, config_path: str, checkpoints: str, wandb: bool) -> None:
    try:
        import lightning as L
    except ImportError:
        raise click.ClickException("pip install -e '.[train]' first")

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    from spinai_ocr.training.loop import (
        DetectionModule,
        OptimCfg,
        RecognitionModule,
    )

    optim_cfg = OptimCfg(
        lr=float(cfg["train"].get("lr", 1e-3)),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
        optimizer=cfg["train"].get("optimizer", "adamw"),
        scheduler=cfg["train"].get("scheduler", "cosine"),
        warmup_steps=int(cfg["train"].get("warmup_steps", 2000)),
        total_steps=int(cfg["train"].get("total_steps", 100_000)),
    )

    if task == "detection":
        module = DetectionModule(
            backbone=cfg["model"].get("backbone", "resnet18"),
            optim_cfg=optim_cfg,
        )
    else:
        from spinai_ocr.vocab.base import load_vocab

        vocab = load_vocab(cfg["model"].get("vocab", "ko_en_v1"))
        module = RecognitionModule(
            arch=cfg["model"].get("arch", "svtr_lite"),
            vocab_size=vocab.size,
            input_height=cfg["model"].get("input_height", 48),
            optim_cfg=optim_cfg,
        )

    ckpt_dir = Path(checkpoints) / cfg.get("name", task)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        L.pytorch.callbacks.ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="{epoch:02d}-{val/total:.4f}" if task == "detection" else "{epoch:02d}-{val/ctc:.4f}",
            save_top_k=3,
            monitor="val/total" if task == "detection" else "val/ctc",
            mode="min",
        ),
        L.pytorch.callbacks.LearningRateMonitor(logging_interval="step"),
    ]
    logger = False
    if wandb:
        try:
            from lightning.pytorch.loggers import WandbLogger  # type: ignore

            logger = WandbLogger(project=cfg.get("log", {}).get("project", "spinai-ocr"))
        except ImportError:
            click.echo("[warn] wandb not installed; proceeding without logger")

    trainer = L.Trainer(
        max_epochs=int(cfg["train"].get("epochs", 40)),
        precision=cfg["train"].get("precision", "bf16-mixed"),
        gradient_clip_val=float(cfg["train"].get("grad_clip", 1.0)),
        default_root_dir=str(ckpt_dir),
        callbacks=callbacks,
        logger=logger,
    )

    # Datamodule wiring is project-specific: the user should set up a
    # LightningDataModule that yields {"labeled": {...}, "unlabeled": {...}}.
    click.echo("Trainer constructed. Wire a LightningDataModule and call trainer.fit(module, dm).")
    click.echo(f"Module class: {module.__class__.__name__}")


if __name__ == "__main__":
    main()
