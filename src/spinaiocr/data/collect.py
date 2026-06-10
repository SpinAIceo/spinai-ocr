"""Data collection orchestrator.

Downloads public OCR datasets into `data/raw/<source_name>/` and writes a
manifest JSON with license info so downstream code can filter by license.

Usage:
    spinai-ocr-collect --source all --lang ko --out data/raw
    spinai-ocr-collect --source hf --lang ko,en --out data/raw
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import click

from spinaiocr.data.sources import SOURCES, DataSource, filter_sources

MANIFEST_NAME = "manifest.json"


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")


def _write_manifest(out_dir: Path, entries: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "count": len(entries),
        "entries": entries,
    }
    (out_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _download_kaggle(slug: str, dest: Path) -> bool:
    """Download via kaggle CLI. Requires `kaggle` installed + kaggle.json set up."""
    dest.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"],
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        click.echo(f"[kaggle] {slug} failed: {e}", err=True)
        return False


def _download_hf(dataset_id: str, dest: Path) -> bool:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        click.echo("[hf] huggingface-hub not installed. pip install -e '.[data]'", err=True)
        return False
    try:
        snapshot_download(
            repo_id=dataset_id,
            repo_type="dataset",
            local_dir=str(dest),
            local_dir_use_symlinks=False,
        )
        return True
    except Exception as e:  # noqa: BLE001
        click.echo(f"[hf] {dataset_id} failed: {e}", err=True)
        return False


def _download_http(url: str, dest: Path) -> bool:
    """Placeholder — real HTTP downloads often need manual login (AI Hub, ICDAR).

    This records the URL so a human operator can fetch it; we intentionally do
    NOT scrape behind login walls automatically.
    """
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "SOURCE_URL.txt").write_text(url, encoding="utf-8")
    (dest / "README.txt").write_text(
        f"Manual download required.\nVisit: {url}\n"
        f"Place extracted files in this directory, then re-run pipeline.\n",
        encoding="utf-8",
    )
    return True


def _download_github(identifier: str, dest: Path) -> bool:
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", f"https://github.com/{identifier}", str(dest)],
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        click.echo(f"[github] {identifier} failed: {e}", err=True)
        return False


_DISPATCH = {
    "kaggle": lambda s, d: _download_kaggle(s.identifier, d),
    "hf": lambda s, d: _download_hf(s.identifier, d),
    "github": lambda s, d: _download_github(s.identifier, d),
    "http": lambda s, d: _download_http(s.url, d),
    "icdar": lambda s, d: _download_http(s.url, d),
    "aihub": lambda s, d: _download_http(s.url, d),
    "synthetic": lambda s, d: _download_github(s.identifier, d) if s.identifier else False,
}


def run_collect(
    out_dir: Path,
    lang: str | None = None,
    kind: str | None = None,
    task: str | None = None,
    commercial_only: bool = False,
    dry_run: bool = False,
) -> list[dict]:
    chosen: list[DataSource] = filter_sources(
        lang=lang, task=task, commercial_only=commercial_only
    )
    if kind and kind != "all":
        chosen = [s for s in chosen if s.kind == kind]

    entries = []
    for src in chosen:
        dest = out_dir / _slug(src.name)
        click.echo(f"→ {src.name} [{src.kind}] → {dest}")
        if dry_run:
            ok = True
        else:
            fn = _DISPATCH.get(src.kind)
            ok = bool(fn and fn(src, dest))
        entries.append(
            {
                **asdict(src),
                "local_path": str(dest),
                "downloaded": ok,
            }
        )
    _write_manifest(out_dir, entries)
    return entries


@click.command(context_settings={"show_default": True})
@click.option("--source", "kind", default="all", help="Source kind: kaggle|hf|github|http|icdar|aihub|synthetic|all")
@click.option("--lang", default=None, help="Language filter: ko, en, ja, zh, ...")
@click.option("--task", default=None, help="Task filter: detection|recognition|layout|synthetic")
@click.option("--commercial-only", is_flag=True, help="Only include commercially-usable sources")
@click.option("--out", "out_dir", default="data/raw", type=click.Path(), help="Output root directory")
@click.option("--dry-run", is_flag=True, help="List sources without downloading")
def main(kind: str, lang: str | None, task: str | None, commercial_only: bool, out_dir: str, dry_run: bool) -> None:
    """Collect OCR datasets from registered public sources."""
    out = Path(out_dir)
    entries = run_collect(
        out_dir=out,
        lang=lang,
        kind=None if kind == "all" else kind,
        task=task,
        commercial_only=commercial_only,
        dry_run=dry_run,
    )
    ok = sum(1 for e in entries if e["downloaded"])
    click.echo(f"\nDone. {ok}/{len(entries)} succeeded. Manifest: {out / MANIFEST_NAME}")
    if ok < len(entries):
        sys.exit(1)


if __name__ == "__main__":
    main()
