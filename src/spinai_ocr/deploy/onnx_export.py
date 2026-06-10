"""Export trained checkpoints to ONNX.

Example:
    python -m spinai_ocr.deploy.onnx_export \
        --task detection --ckpt checkpoints/det.pth \
        --out checkpoints/det.onnx --opset 17
"""
from __future__ import annotations

from pathlib import Path

import click
import torch

from spinai_ocr.models.detection import DBNet
from spinai_ocr.models.recognition import build_recognition
from spinai_ocr.vocab.base import load_vocab


def _dummy_input(task: str, input_size: int, height: int) -> torch.Tensor:
    if task == "detection":
        return torch.randn(1, 3, input_size, input_size)
    return torch.randn(1, 3, height, 4 * height)


def _build(task: str, arch: str, vocab: str, input_height: int) -> torch.nn.Module:
    if task == "detection":
        return DBNet(backbone=arch)
    v = load_vocab(vocab)
    return build_recognition(arch, vocab_size=v.size, input_height=input_height)


def _load_state(model: torch.nn.Module, ckpt: Path) -> None:
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict):
        if "state_dict" in state:
            state = state["state_dict"]
        # strip 'student.' prefix from Lightning MeanTeacher modules
        state = {k.replace("student.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)


@click.command()
@click.option("--task", type=click.Choice(["detection", "recognition"]), required=True)
@click.option("--ckpt", required=True, type=click.Path(exists=True))
@click.option("--out", "out_path", required=True, type=click.Path())
@click.option("--arch", default=None, help="detection backbone or recognition arch")
@click.option("--vocab", default="ko_en_v1")
@click.option("--input-size", default=960, type=int, help="detection square input")
@click.option("--input-height", default=48, type=int, help="recognition height")
@click.option("--opset", default=17, type=int)
@click.option("--fp16", is_flag=True, help="cast to fp16 before export")
@click.option("--quantize", is_flag=True, help="int8 dynamic quantization post-export")
def main(
    task: str,
    ckpt: str,
    out_path: str,
    arch: str | None,
    vocab: str,
    input_size: int,
    input_height: int,
    opset: int,
    fp16: bool,
    quantize: bool,
) -> None:
    arch = arch or ("resnet18" if task == "detection" else "svtr_lite")
    model = _build(task, arch, vocab, input_height)
    _load_state(model, Path(ckpt))
    model.eval()
    if fp16:
        model = model.half()

    example = _dummy_input(task, input_size, input_height)
    if fp16:
        example = example.half()

    dyn = {"image": {0: "batch", 2: "height", 3: "width"}, "output": {0: "batch"}}
    torch.onnx.export(
        model,
        example,
        out_path,
        input_names=["image"],
        output_names=["output"] if task != "detection" else ["prob", "thresh", "binary"],
        opset_version=opset,
        dynamic_axes=dyn,
    )
    click.echo(f"Wrote {out_path}")

    if quantize:
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic  # type: ignore
        except ImportError:
            raise click.ClickException("pip install onnxruntime for quantization")
        q_path = str(Path(out_path).with_suffix(".int8.onnx"))
        quantize_dynamic(out_path, q_path, weight_type=QuantType.QInt8)
        click.echo(f"Wrote {q_path}")


if __name__ == "__main__":
    main()
