"""Fail-closed helpers for analyses whose inputs and outputs must stay out of Git.

The helpers are intentionally generic: no instrument, account, or allocation data
belongs in this module or its tests.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd


class PrivatePathError(ValueError):
    """A requested private path is unsafe or insufficiently restricted."""


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _permission_bits(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _reject_symlink_components(path: str | Path, role: str) -> None:
    """Reject a symlink in any existing component before canonicalization."""

    absolute = Path(path).expanduser().absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise PrivatePathError(f"private {role} path contains a symbolic link")


def validate_private_paths(
    *,
    input_path: str | Path,
    output_directory: str | Path,
    forbidden_roots: Sequence[str | Path],
) -> tuple[Path, Path]:
    """Validate private input/output locations before reading sensitive bytes.

    Both locations must be outside every supplied Git/workspace root, must not be
    symlinks, and must deny all group/other permissions. The caller creates the
    output directory explicitly; this function never guesses a destination.
    """

    if not forbidden_roots:
        raise PrivatePathError("at least one forbidden root is required")
    source_argument = Path(input_path).expanduser()
    destination_argument = Path(output_directory).expanduser()
    _reject_symlink_components(source_argument, "input")
    _reject_symlink_components(destination_argument, "output")
    source = source_argument.resolve()
    destination = destination_argument.resolve()
    if not source.is_file():
        raise PrivatePathError("private input must be an existing regular file")
    if not destination.is_dir():
        raise PrivatePathError("private output must be an existing directory")
    roots = tuple(_resolved(root) for root in forbidden_roots)
    for path, role in ((source, "input"), (destination, "output")):
        if any(_is_within(path, root) for root in roots):
            raise PrivatePathError(f"private {role} path is inside a forbidden root")
        if _permission_bits(path) & 0o077:
            raise PrivatePathError(
                f"private {role} permissions must deny all group/other access"
            )
    if _permission_bits(source.parent) & 0o077:
        raise PrivatePathError("private input parent must deny group/other access")
    return source, destination


def _atomic_private_file(destination: Path, writer: Any) -> Path:
    if destination.is_symlink():
        raise PrivatePathError("private output must not be a symbolic link")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.chmod(temporary_name, 0o600)
    try:
        writer(descriptor, Path(temporary_name))
        os.replace(temporary_name, destination)
        os.chmod(destination, 0o600)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


def _validated_output_path(
    path: str | Path,
    *,
    private_root: str | Path,
    forbidden_roots: Sequence[str | Path],
) -> Path:
    """Resolve one output beneath an external restricted root, without symlinks."""

    if not forbidden_roots:
        raise PrivatePathError("at least one forbidden root is required")
    root_argument = Path(private_root).expanduser().absolute()
    destination_argument = Path(path).expanduser().absolute()
    _reject_symlink_components(root_argument, "output")
    _reject_symlink_components(destination_argument, "output")
    if not root_argument.is_dir():
        raise PrivatePathError("private output root must be an existing real directory")
    if _permission_bits(root_argument) & 0o077:
        raise PrivatePathError("private output root must deny all group/other access")
    try:
        relative = destination_argument.relative_to(root_argument)
    except ValueError as exc:
        raise PrivatePathError("private output must be beneath the validated root") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise PrivatePathError("private output must be a non-traversing file path")

    parent = root_argument
    for part in relative.parts[:-1]:
        parent = parent / part
        if parent.is_symlink():
            raise PrivatePathError("private output parents must not be symbolic links")
        parent.mkdir(mode=0o700, exist_ok=True)
        if not parent.is_dir() or _permission_bits(parent) & 0o077:
            raise PrivatePathError(
                "private output parents must be restricted real directories"
            )
    if destination_argument.is_symlink():
        raise PrivatePathError("private output must not be a symbolic link")

    root = root_argument.resolve(strict=True)
    destination = destination_argument.resolve(strict=False)
    if not _is_within(destination, root):
        raise PrivatePathError("private output escapes its validated root")
    for forbidden in forbidden_roots:
        forbidden_root = _resolved(forbidden)
        if _is_within(root, forbidden_root) or _is_within(destination, forbidden_root):
            raise PrivatePathError("private output is inside a forbidden root")
    return destination


def atomic_private_json(
    path: str | Path,
    value: Mapping[str, Any],
    *,
    private_root: str | Path,
    forbidden_roots: Sequence[str | Path],
) -> Path:
    """Write JSON atomically with mode 600 and no logging side effects."""

    destination = _validated_output_path(
        path, private_root=private_root, forbidden_roots=forbidden_roots
    )

    def writer(descriptor: int, _temporary: Path) -> None:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    return _atomic_private_file(destination, writer)


def atomic_private_text(
    path: str | Path,
    value: str,
    *,
    private_root: str | Path,
    forbidden_roots: Sequence[str | Path],
) -> Path:
    """Write UTF-8 text atomically with mode 600."""

    destination = _validated_output_path(
        path, private_root=private_root, forbidden_roots=forbidden_roots
    )

    def writer(descriptor: int, _temporary: Path) -> None:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())

    return _atomic_private_file(destination, writer)


def atomic_private_parquet(
    path: str | Path,
    frame: pd.DataFrame,
    *,
    private_root: str | Path,
    forbidden_roots: Sequence[str | Path],
) -> Path:
    """Write a Parquet frame atomically with mode 600."""

    destination = _validated_output_path(
        path, private_root=private_root, forbidden_roots=forbidden_roots
    )

    def writer(descriptor: int, temporary: Path) -> None:
        os.close(descriptor)
        frame.to_parquet(temporary, index=False)

    return _atomic_private_file(destination, writer)


__all__ = [
    "PrivatePathError",
    "atomic_private_json",
    "atomic_private_parquet",
    "atomic_private_text",
    "validate_private_paths",
]
