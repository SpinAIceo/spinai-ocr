"""Logged git operations.

Wraps `git add`, `git commit`, `git push` so every step ends up in the
structured log. Confirms before destructive ops (force push) at the caller
level — this module itself does not prompt.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from spinai_ocr.log import capture_crashes, get_logger, log_span

log = get_logger("spinai_ocr.deploy.git")


def _git(cmd: list[str], cwd: str | Path | None = None) -> tuple[int, str]:
    full = ["git", *cmd]
    log.debug("git.exec %s cwd=%s", " ".join(full), cwd,
              extra={"op": "git", "cmd": full, "cwd": str(cwd) if cwd else None})
    try:
        proc = subprocess.run(
            full, cwd=str(cwd) if cwd else None,
            check=False, capture_output=True, text=True,
        )
    except FileNotFoundError:
        log.error("git.not_found install git and retry")
        return 127, ""
    out = (proc.stdout or "") + (proc.stderr or "")
    level = log.info if proc.returncode == 0 else log.error
    level("git.result cmd=%s rc=%d out=%s", " ".join(cmd), proc.returncode,
          out.strip().splitlines()[-1] if out.strip() else "",
          extra={"op": "git", "cmd": full, "rc": proc.returncode})
    return proc.returncode, out


def git_status(cwd: str | Path | None = None) -> str:
    rc, out = _git(["status", "--porcelain"], cwd=cwd)
    if rc != 0:
        raise RuntimeError(f"git status failed rc={rc}")
    changed = [ln for ln in out.splitlines() if ln.strip()]
    log.info("git.status n_changes=%d", len(changed),
             extra={"op": "git.status", "n_changes": len(changed)})
    return out


def git_add(paths: list[str | Path], cwd: str | Path | None = None) -> None:
    paths = [str(p) for p in paths]
    with capture_crashes("git.add", extra={"paths": paths}):
        with log_span("git.add", n_paths=len(paths)):
            rc, _ = _git(["add", *paths], cwd=cwd)
            if rc != 0:
                raise RuntimeError(f"git add failed rc={rc}")


def git_commit(message: str, cwd: str | Path | None = None, allow_empty: bool = False) -> str:
    """Create a commit. Returns the new commit SHA."""
    with capture_crashes("git.commit", extra={"message": message[:120]}):
        with log_span("git.commit"):
            cmd = ["commit", "-m", message]
            if allow_empty:
                cmd.append("--allow-empty")
            rc, out = _git(cmd, cwd=cwd)
            if rc != 0:
                raise RuntimeError(f"git commit failed rc={rc}\n{out}")
            rc, sha = _git(["rev-parse", "HEAD"], cwd=cwd)
            sha = sha.strip()
            log.info("git.commit.committed sha=%s message=%r", sha[:8], message[:60],
                     extra={"op": "git.commit", "sha": sha, "event": "committed"})
            return sha


def git_push(remote: str = "origin", branch: str | None = None, *, force: bool = False,
             cwd: str | Path | None = None) -> None:
    if force:
        log.warning("git.push.force FORCE PUSH to %s %s — destructive!", remote, branch or "",
                    extra={"op": "git.push", "force": True, "remote": remote, "branch": branch})
    with capture_crashes("git.push", extra={"remote": remote, "branch": branch, "force": force}):
        with log_span("git.push", remote=remote, branch=branch or "?"):
            cmd = ["push"]
            if force:
                cmd.append("--force-with-lease")
            cmd.append(remote)
            if branch:
                cmd.append(branch)
            rc, out = _git(cmd, cwd=cwd)
            if rc != 0:
                raise RuntimeError(f"git push failed rc={rc}\n{out}")
            log.info("git.push.pushed remote=%s branch=%s", remote, branch or "",
                     extra={"op": "git.push", "event": "pushed"})


def current_sha(cwd: str | Path | None = None) -> str:
    rc, out = _git(["rev-parse", "HEAD"], cwd=cwd)
    if rc != 0:
        raise RuntimeError(f"git rev-parse failed rc={rc}")
    return out.strip()


def current_branch(cwd: str | Path | None = None) -> str:
    rc, out = _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd)
    if rc != 0:
        return "(detached)"
    return out.strip()
