"""Top-level CLI: `spinai-ocr`.

Subcommands:
    run        — run OCR on image(s) or a folder
    info       — print version / config info
    collect    — alias to `spinai-ocr-collect`
    bench      — alias to `spinai-ocr-bench`
"""
from __future__ import annotations

import json
from pathlib import Path

import click

from spinaiocr import __version__
from spinaiocr.config import PipelineConfig
from spinaiocr.inference.pipeline import OCRPipeline


@click.group()
@click.version_option(__version__, prog_name="spinai-ocr")
def main() -> None:
    """SPINAI OCR — Korean-first open-source OCR engine."""


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--lang", default="ko", help="Language code")
@click.option("--tier", default="lite", type=click.Choice(["lite", "standard", "large"]))
@click.option("--decode", default="beam_lm", type=click.Choice(["greedy", "beam_lm"]),
              help="Decoder. greedy ~7ms, beam_lm ~25ms (more accurate).")
@click.option("--single-line/--multi-line", default=None,
              help="Force single-line crop (skip detection). Default: auto by aspect.")
@click.option("--show-confidence", is_flag=True,
              help="Append per-line confidence (only affects --format text).")
@click.option("--output", "output_path", default=None, type=click.Path())
@click.option("--format", "out_fmt", default="text", type=click.Choice(["text", "json"]))
def run(path: str, lang: str, tier: str, decode: str,
        single_line: bool | None, show_confidence: bool,
        output_path: str | None, out_fmt: str) -> None:
    """Run OCR on a single image or all images under a folder."""
    cfg = PipelineConfig(lang=lang, tier=tier)
    pipe = OCRPipeline(config=cfg)

    p = Path(path)
    images = [p] if p.is_file() else sorted(
        [x for x in p.rglob("*") if x.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}]
    )

    results = []
    for img in images:
        res = pipe(str(img), single_line=single_line, decode_mode=decode)
        results.append(
            {
                "image": str(img),
                "lines": [
                    {"text": l.text, "bbox": l.bbox, "confidence": l.confidence}
                    for l in res.lines
                ],
            }
        )

    if out_fmt == "json":
        payload = json.dumps(results, ensure_ascii=False, indent=2)
    else:
        lines = []
        for r in results:
            lines.append(f"=== {r['image']} ===")
            for l in r["lines"]:
                if show_confidence:
                    lines.append(f"[{l['confidence'] * 100:5.1f}%] {l['text']}")
                else:
                    lines.append(l["text"])
        payload = "\n".join(lines)

    if output_path:
        Path(output_path).write_text(payload, encoding="utf-8")
        click.echo(f"Wrote {output_path}")
    else:
        click.echo(payload)


@main.command()
def info() -> None:
    """Print version and default config."""
    click.echo(f"SPINAI OCR v{__version__}")
    click.echo(PipelineConfig().model_dump_json(indent=2))


if __name__ == "__main__":
    main()
