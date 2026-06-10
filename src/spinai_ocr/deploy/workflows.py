"""Deployment, upload, and build workflows with step-by-step logs.

Every function here logs:
    * op.begin   — what's about to happen, with inputs
    * op.step    — each sub-step (per file, per layer, per bytes chunk)
    * op.commit  — what was produced, size, sha, elapsed_ms
    * op.failed  — traceback + crash dump on any failure

Functions:
    build_docker_image()    — wraps `docker build`, streams output
    push_docker_image()     — wraps `docker push`, streams output
    export_onnx_model()     — wraps deploy/onnx_export.py with logging
    upload_to_huggingface() — upload a directory of checkpoints to HF Hub
    start_vllm_server()     — `docker compose up` with health-check wait
    run_pip_install()       — pip install with streamed output
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Iterable

from spinai_ocr.io.logged_io import sha256_of_file
from spinai_ocr.log import capture_crashes, get_logger, log_span

log = get_logger("spinai_ocr.deploy")


def _stream_process(
    cmd: list[str],
    *,
    cwd: str | Path | None = None,
    env: dict | None = None,
    op_name: str,
) -> int:
    """Run a subprocess and log stdout/stderr line-by-line.

    Returns the process return code. Does NOT raise on non-zero — caller decides.
    """
    log.info("%s.exec cmd=%s cwd=%s", op_name, " ".join(cmd), cwd,
             extra={"op": op_name, "cmd": cmd, "cwd": str(cwd) if cwd else None})
    proc_env = os.environ.copy()
    if env:
        proc_env.update(env)

    started = time.perf_counter()
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=proc_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError as e:
        log.error("%s.exec_failed cmd=%s missing_binary=%s", op_name, cmd, e.filename,
                  extra={"op": op_name, "missing_binary": e.filename})
        return 127

    n_lines = 0
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        n_lines += 1
        # Classify a few common patterns so they show up in log searches.
        lvl = "info"
        if "error" in line.lower() or "fatal" in line.lower():
            lvl = "error"
        elif "warn" in line.lower() or "deprecat" in line.lower():
            lvl = "warning"
        getattr(log, lvl)("%s.out %s", op_name, line,
                          extra={"op": op_name, "stream": "stdout"})
    rc = proc.wait()
    elapsed_ms = (time.perf_counter() - started) * 1000
    log.info(
        "%s.exit rc=%d elapsed_ms=%.1f n_output_lines=%d",
        op_name, rc, elapsed_ms, n_lines,
        extra={"op": op_name, "rc": rc, "elapsed_ms": elapsed_ms, "n_output_lines": n_lines},
    )
    return rc


# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------


def build_docker_image(tag: str, dockerfile: str | Path = "Dockerfile", context: str | Path = ".") -> int:
    """Wraps `docker build -t <tag> -f <dockerfile> <context>`."""
    with capture_crashes("docker.build", extra={"tag": tag, "dockerfile": str(dockerfile)}):
        with log_span("docker.build", tag=tag, dockerfile=str(dockerfile)):
            rc = _stream_process(
                ["docker", "build", "-t", tag, "-f", str(dockerfile), str(context)],
                op_name="docker.build",
            )
            if rc != 0:
                raise RuntimeError(f"docker build failed rc={rc} tag={tag}")
            return rc


def push_docker_image(tag: str) -> int:
    """Wraps `docker push <tag>`. Warns destructively for main/master."""
    if tag.endswith(":latest") or tag.endswith(":main"):
        log.warning("docker.push.warn pushing mutable tag=%s — consider pinning a SHA", tag,
                    extra={"op": "docker.push", "tag": tag})
    with capture_crashes("docker.push", extra={"tag": tag}):
        with log_span("docker.push", tag=tag):
            rc = _stream_process(["docker", "push", tag], op_name="docker.push")
            if rc != 0:
                raise RuntimeError(f"docker push failed rc={rc} tag={tag}")
            return rc


# ---------------------------------------------------------------------------
# vLLM
# ---------------------------------------------------------------------------


def start_vllm_server(compose_dir: str | Path = "deploy/vllm", wait_healthy: float = 120.0) -> None:
    """`docker compose up -d` + poll /health until ready."""
    compose_dir = Path(compose_dir)
    with capture_crashes("vllm.start", extra={"compose_dir": str(compose_dir)}):
        with log_span("vllm.start"):
            rc = _stream_process(
                ["docker", "compose", "up", "-d"],
                cwd=compose_dir,
                op_name="vllm.compose_up",
            )
            if rc != 0:
                raise RuntimeError("docker compose up failed")

            import urllib.error
            import urllib.request
            deadline = time.perf_counter() + wait_healthy
            while time.perf_counter() < deadline:
                try:
                    urllib.request.urlopen("http://localhost:8000/health", timeout=2)
                    log.info("vllm.ready", extra={"op": "vllm.start", "event": "ready"})
                    return
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(2)
            raise TimeoutError(f"vLLM did not become healthy within {wait_healthy}s")


# ---------------------------------------------------------------------------
# ONNX export (wrapper with logging around the existing CLI)
# ---------------------------------------------------------------------------


def export_onnx_model(
    task: str,
    ckpt: str | Path,
    out_path: str | Path,
    *,
    arch: str | None = None,
    vocab: str = "ko_en_v1",
    input_size: int = 960,
    input_height: int = 48,
    opset: int = 17,
    fp16: bool = False,
    quantize: bool = False,
) -> Path:
    ckpt = Path(ckpt)
    out_path = Path(out_path)
    with capture_crashes("onnx.export",
                         extra={"task": task, "ckpt": str(ckpt), "out": str(out_path)}):
        with log_span("onnx.export", task=task, out=str(out_path)):
            cmd = [
                "python", "-m", "spinai_ocr.deploy.onnx_export",
                "--task", task, "--ckpt", str(ckpt),
                "--out", str(out_path), "--vocab", vocab,
                "--input-size", str(input_size), "--input-height", str(input_height),
                "--opset", str(opset),
            ]
            if arch:
                cmd += ["--arch", arch]
            if fp16:
                cmd.append("--fp16")
            if quantize:
                cmd.append("--quantize")
            rc = _stream_process(cmd, op_name="onnx.export")
            if rc != 0:
                raise RuntimeError(f"onnx export failed rc={rc}")
            size = out_path.stat().st_size
            sha = sha256_of_file(out_path)[:12]
            log.info("onnx.artifact path=%s size_MB=%.2f sha256=%s...",
                     out_path, size / 1e6, sha,
                     extra={"op": "onnx.export", "path": str(out_path),
                            "size_bytes": size, "sha256_prefix": sha})
            return out_path


# ---------------------------------------------------------------------------
# HuggingFace Hub upload
# ---------------------------------------------------------------------------


def upload_to_huggingface(
    local_dir: str | Path,
    repo_id: str,
    *,
    repo_type: str = "model",
    path_in_repo: str = "",
    commit_message: str | None = None,
    private: bool = False,
) -> None:
    """Uploads a local directory to the HF Hub with per-file progress logs."""
    local_dir = Path(local_dir)
    if not local_dir.exists():
        raise FileNotFoundError(local_dir)

    files = [p for p in local_dir.rglob("*") if p.is_file()]
    total_bytes = sum(p.stat().st_size for p in files)
    log.info(
        "hf.upload.begin repo=%s type=%s n_files=%d total=%s private=%s",
        repo_id, repo_type, len(files), _fmt_bytes(total_bytes), private,
        extra={"op": "hf.upload", "repo_id": repo_id, "repo_type": repo_type,
               "n_files": len(files), "total_bytes": total_bytes, "private": private},
    )

    try:
        from huggingface_hub import HfApi, create_repo  # type: ignore
    except ImportError as e:
        log.error("hf.upload.missing_dep install huggingface-hub first: %s", e,
                  extra={"op": "hf.upload"})
        raise

    api = HfApi()
    with capture_crashes("hf.upload",
                         extra={"repo_id": repo_id, "local_dir": str(local_dir)}):
        create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True)
        log.info("hf.upload.repo_ready repo=%s", repo_id,
                 extra={"op": "hf.upload", "repo_id": repo_id, "event": "repo_ready"})

        uploaded = 0
        started = time.perf_counter()
        for p in sorted(files):
            rel = p.relative_to(local_dir).as_posix()
            size = p.stat().st_size
            log.debug("hf.upload.file path=%s size=%s", rel, _fmt_bytes(size),
                      extra={"op": "hf.upload", "rel_path": rel, "size_bytes": size})
            api.upload_file(
                path_or_fileobj=str(p),
                path_in_repo=f"{path_in_repo}/{rel}".lstrip("/"),
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=commit_message,
            )
            uploaded += size
            pct = uploaded / total_bytes * 100 if total_bytes else 100
            log.info("hf.upload.progress %.1f%% %s/%s",
                     pct, _fmt_bytes(uploaded), _fmt_bytes(total_bytes),
                     extra={"op": "hf.upload", "pct": pct, "uploaded": uploaded,
                            "total": total_bytes})
        elapsed = time.perf_counter() - started
        log.info(
            "hf.upload.commit repo=%s files=%d bytes=%s elapsed_s=%.1f speed=%.2f MB/s",
            repo_id, len(files), _fmt_bytes(total_bytes), elapsed,
            (total_bytes / 1e6) / max(elapsed, 1e-6),
            extra={"op": "hf.upload", "event": "commit",
                   "repo_id": repo_id, "elapsed_s": elapsed},
        )


# ---------------------------------------------------------------------------
# pip
# ---------------------------------------------------------------------------


def run_pip_install(packages: list[str] | str, *, upgrade: bool = False, editable: str | Path | None = None) -> int:
    cmd = ["pip", "install"]
    if upgrade:
        cmd.append("--upgrade")
    if editable:
        cmd += ["-e", str(editable)]
    else:
        if isinstance(packages, str):
            cmd.append(packages)
        else:
            cmd += list(packages)
    with capture_crashes("pip.install", extra={"cmd": cmd}):
        with log_span("pip.install"):
            rc = _stream_process(cmd, op_name="pip.install")
            if rc != 0:
                raise RuntimeError(f"pip install failed rc={rc}")
            return rc


def _fmt_bytes(n) -> str:
    if n is None:
        return "?"
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.2f}{unit}" if unit != "B" else f"{int(n)}B"
        n /= 1024
    return f"{n:.2f}PB"
