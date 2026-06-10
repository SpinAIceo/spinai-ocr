"""`spinai-ocr-logs` — inspect structured logs and crash dumps.

Subcommands::
    tail        stream the text log (human)
    errors      show only ERROR/CRITICAL records from the JSONL
    crashes     list crash dumps (latest first) + pretty-print one
    stats       aggregate by logger name / level / last N minutes
    spans       slowest spans by elapsed_ms
    nanfind     find NaN/Inf events
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import click


def _log_dir() -> Path:
    import os

    return Path(os.environ.get("SPINAI_LOG_DIR", "logs"))


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _color(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m"


LEVEL_COLORS = {
    "DEBUG": "38;5;109",
    "INFO": "38;5;114",
    "WARNING": "38;5;214",
    "ERROR": "38;5;203",
    "CRITICAL": "48;5;124;97",
}


@click.group()
@click.option("--log-dir", default=None, type=click.Path(), help="override SPINAI_LOG_DIR")
@click.pass_context
def main(ctx, log_dir):
    """Inspect SPINAI OCR logs."""
    ctx.ensure_object(dict)
    ctx.obj["dir"] = Path(log_dir) if log_dir else _log_dir()


@main.command()
@click.option("--lines", "-n", default=40)
@click.option("--follow", "-f", is_flag=True, help="tail -f style")
@click.pass_context
def tail(ctx, lines, follow):
    """Tail the human-readable log."""
    path = ctx.obj["dir"] / "spinai.log"
    if not path.exists():
        raise click.ClickException(f"no log at {path}")
    with path.open("r", encoding="utf-8") as f:
        content = f.read().splitlines()
    for ln in content[-lines:]:
        click.echo(ln)
    if follow:
        pos = path.stat().st_size
        while True:
            time.sleep(0.5)
            with path.open("r", encoding="utf-8") as f:
                f.seek(pos)
                new = f.read()
            if new:
                click.echo(new, nl=False)
                pos += len(new.encode("utf-8"))


@main.command()
@click.option("--min-level", default="ERROR", type=click.Choice(["WARNING", "ERROR", "CRITICAL"]))
@click.option("--since-minutes", default=60, type=int)
@click.option("--limit", default=50, type=int)
@click.pass_context
def errors(ctx, min_level, since_minutes, limit):
    """Show recent errors from JSONL."""
    path = ctx.obj["dir"] / "spinai.jsonl"
    cutoff = time.time() - since_minutes * 60
    target = {"WARNING": 30, "ERROR": 40, "CRITICAL": 50}[min_level]
    levels = {"WARNING": 30, "ERROR": 40, "CRITICAL": 50, "DEBUG": 10, "INFO": 20}
    records = [r for r in _iter_jsonl(path)
               if levels.get(r.get("level", ""), 0) >= target
               and r.get("ts", 0) >= cutoff]
    records.sort(key=lambda r: r.get("ts", 0), reverse=True)
    for r in records[:limit]:
        ts = time.strftime("%H:%M:%S", time.localtime(r.get("ts", 0)))
        lvl = r.get("level", "?")
        lvl_col = _color(f"{lvl:<8}", LEVEL_COLORS.get(lvl, "0"))
        click.echo(
            f"{ts} {lvl_col} {r.get('logger', '?')}:{r.get('line', '?')}  {r.get('msg', '')}"
        )
        exc = r.get("exc")
        if exc:
            click.echo(_color(f"        {exc.get('type', '?')}: {exc.get('message', '')}", "2"))
        extra = r.get("extra") or {}
        for k, v in extra.items():
            if k in {"span", "event", "elapsed_ms"}:
                continue
            click.echo(_color(f"        {k}={v}", "2"))


@main.command()
@click.option("--show", default=None, help="dump id to pretty-print")
@click.option("--limit", default=20, type=int)
@click.pass_context
def crashes(ctx, show, limit):
    """List or show crash dumps."""
    crash_dir = ctx.obj["dir"] / "crashes"
    if not crash_dir.exists():
        click.echo(f"no crash dir at {crash_dir}")
        return
    files = sorted(crash_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if show:
        match = next((p for p in files if show in p.name), None)
        if not match:
            raise click.ClickException(f"no dump matching '{show}'")
        payload = json.loads(match.read_text(encoding="utf-8"))
        click.echo(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    for p in files[:limit]:
        data = json.loads(p.read_text(encoding="utf-8"))
        exc = data.get("exception", {})
        click.echo(f"{p.name}  {exc.get('type', '?')}: {exc.get('message', '')}")


@main.command()
@click.option("--since-minutes", default=60, type=int)
@click.pass_context
def stats(ctx, since_minutes):
    """Per-logger / per-level counts over the last window."""
    path = ctx.obj["dir"] / "spinai.jsonl"
    cutoff = time.time() - since_minutes * 60
    by_level: Counter = Counter()
    by_logger: Counter = Counter()
    for r in _iter_jsonl(path):
        if r.get("ts", 0) < cutoff:
            continue
        by_level[r.get("level", "?")] += 1
        by_logger[r.get("logger", "?")] += 1
    click.echo("-- level counts --")
    for lvl, c in by_level.most_common():
        click.echo(f"  {lvl:<10} {c}")
    click.echo("\n-- top loggers --")
    for name, c in by_logger.most_common(15):
        click.echo(f"  {c:>6}  {name}")


@main.command()
@click.option("--top", default=20, type=int)
@click.pass_context
def spans(ctx, top):
    """Slowest spans by elapsed_ms."""
    path = ctx.obj["dir"] / "spinai.jsonl"
    slow: list[tuple[float, str, str]] = []
    for r in _iter_jsonl(path):
        extra = r.get("extra") or {}
        ms = extra.get("elapsed_ms")
        if extra.get("event") == "end" and isinstance(ms, (int, float)):
            slow.append((float(ms), extra.get("span", "?"), r.get("logger", "?")))
    slow.sort(reverse=True)
    for ms, name, logger in slow[:top]:
        click.echo(f"  {ms:>10.1f} ms   {name:<30}  {logger}")


@main.command()
@click.option("--since-minutes", default=1440, type=int)
@click.pass_context
def nanfind(ctx, since_minutes):
    """Hunt for NaN/Inf guard events."""
    path = ctx.obj["dir"] / "spinai.jsonl"
    cutoff = time.time() - since_minutes * 60
    pat = re.compile(r"non-finite|NaNInfGuard", re.I)
    for r in _iter_jsonl(path):
        if r.get("ts", 0) < cutoff:
            continue
        if pat.search(r.get("msg", "")):
            ts = time.strftime("%H:%M:%S", time.localtime(r.get("ts", 0)))
            click.echo(f"{ts} {r.get('level'):<8} {r.get('logger')} | {r.get('msg')}")


if __name__ == "__main__":
    main()
