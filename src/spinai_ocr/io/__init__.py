"""Logged I/O primitives.

Every public function here emits structured start/progress/done logs and
surfaces errors with enough context to reproduce. Import these instead of
calling `open()`, `torch.save`, `Path.write_bytes`, etc. directly when you
want visibility into *what was written, how big, where, when, and by whom*.

Design:
    * Short lived: logs `io.begin` + `io.commit` pair around each write.
    * Atomic-ish: writes to `<path>.tmp` then renames to the target. If the
      process dies mid-write the partial file is visible with `.tmp`
      extension, not a half-written real file.
    * Checksums on demand (`sha256=True`) for later verification.
    * Size + duration always logged.
"""
from spinai_ocr.io.logged_io import (
    atomic_write_bytes,
    atomic_write_text,
    download_to_file,
    ensure_dir,
    save_image,
    save_json,
    save_jsonl_append,
    save_torch_ckpt,
    sha256_of_file,
)

__all__ = [
    "atomic_write_bytes",
    "atomic_write_text",
    "download_to_file",
    "ensure_dir",
    "save_image",
    "save_json",
    "save_jsonl_append",
    "save_torch_ckpt",
    "sha256_of_file",
]
