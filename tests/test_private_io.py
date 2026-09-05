from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pandas as pd
import pytest

from leonos.private_io import (
    PrivatePathError,
    atomic_private_json,
    atomic_private_parquet,
    atomic_private_text,
    validate_private_paths,
)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_private_paths_reject_git_root_and_permissive_input(tmp_path: Path) -> None:
    repository = tmp_path / "project"
    private = tmp_path / "restricted"
    repository.mkdir(mode=0o700)
    private.mkdir(mode=0o700)
    inside = repository / "fictional.json"
    inside.write_text("{}", encoding="utf-8")
    os.chmod(inside, 0o600)
    with pytest.raises(PrivatePathError, match="forbidden root"):
        validate_private_paths(
            input_path=inside,
            output_directory=private,
            forbidden_roots=[repository],
        )

    outside = private / "fictional.json"
    outside.write_text("{}", encoding="utf-8")
    os.chmod(outside, 0o644)
    with pytest.raises(PrivatePathError, match="permissions"):
        validate_private_paths(
            input_path=outside,
            output_directory=private,
            forbidden_roots=[repository],
        )

    os.chmod(outside, 0o600)
    link = private / "linked.json"
    link.symlink_to(outside)
    with pytest.raises(PrivatePathError, match="symbolic"):
        validate_private_paths(
            input_path=link,
            output_directory=private,
            forbidden_roots=[repository],
        )


def test_private_atomic_outputs_remain_mode_600(tmp_path: Path) -> None:
    private = tmp_path / "restricted"
    private.mkdir(mode=0o700)
    source = private / "fictional-input.json"
    source.write_text("{}", encoding="utf-8")
    os.chmod(source, 0o600)
    validated_source, validated_directory = validate_private_paths(
        input_path=source,
        output_directory=private,
        forbidden_roots=[tmp_path / "unrelated-repository"],
    )
    assert validated_source == source
    assert validated_directory == private

    forbidden = [tmp_path / "unrelated-repository"]
    json_path = atomic_private_json(
        private / "result.json",
        {"fictional": 1},
        private_root=private,
        forbidden_roots=forbidden,
    )
    text_path = atomic_private_text(
        private / "report.md",
        "fictional\n",
        private_root=private,
        forbidden_roots=forbidden,
    )
    parquet_path = atomic_private_parquet(
        private / "rows.parquet",
        pd.DataFrame({"fictional": [1.0]}),
        private_root=private,
        forbidden_roots=forbidden,
    )
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"fictional": 1}
    assert text_path.read_text(encoding="utf-8") == "fictional\n"
    assert pd.read_parquet(parquet_path).to_dict("list") == {"fictional": [1.0]}
    assert {_mode(json_path), _mode(text_path), _mode(parquet_path)} == {0o600}


def test_private_atomic_outputs_reject_repository_and_symlink_targets(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "project"
    private = tmp_path / "restricted"
    repository.mkdir(mode=0o700)
    private.mkdir(mode=0o700)
    forbidden = [repository]

    with pytest.raises(PrivatePathError, match="at least one forbidden"):
        atomic_private_text(
            private / "unchecked.md",
            "fictional\n",
            private_root=private,
            forbidden_roots=[],
        )

    with pytest.raises(PrivatePathError, match="validated root|forbidden root"):
        atomic_private_text(
            repository / "leak.md",
            "fictional\n",
            private_root=private,
            forbidden_roots=forbidden,
        )

    real = private / "real.md"
    real.write_text("old", encoding="utf-8")
    os.chmod(real, 0o600)
    linked = private / "linked.md"
    linked.symlink_to(real)
    with pytest.raises(PrivatePathError, match="symbolic link"):
        atomic_private_text(
            linked,
            "new",
            private_root=private,
            forbidden_roots=forbidden,
        )
    assert real.read_text(encoding="utf-8") == "old"

    real_parent = private / "real-parent"
    real_parent.mkdir(mode=0o700)
    linked_parent = private / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(PrivatePathError, match="contains a symbolic link"):
        atomic_private_text(
            linked_parent / "nested.md",
            "new",
            private_root=private,
            forbidden_roots=forbidden,
        )
