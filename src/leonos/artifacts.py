"""Reproducible run metadata and atomic small-artifact helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def git_state(repo: str | Path = ".") -> dict[str, Any]:
    """Capture the exact commit and dirty state without pretending dirty is clean."""
    root = Path(repo)

    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
    except subprocess.CalledProcessError:
        commit = None
    status = run("status", "--porcelain=v1")
    return {"commit": commit, "dirty": bool(status), "status": status.splitlines()}


def runtime_environment() -> dict[str, Any]:
    """Return compact, non-secret machine/runtime metadata."""
    return {
        "captured_at": datetime.now(UTC).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pid": os.getpid(),
    }


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(path: str | Path, value: Any) -> Path:
    """Atomically replace a JSON artifact on the same filesystem."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return destination
