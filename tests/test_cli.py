from __future__ import annotations

import pytest

from leonos.cli import build_parser


def test_required_command_surface_parses() -> None:
    parser = build_parser()
    assert parser.parse_args(["data", "fetch"]).data_command == "fetch"
    assert parser.parse_args(["data", "audit"]).data_command == "audit"
    assert parser.parse_args(["prepare"]).command == "prepare"
    assert parser.parse_args(["baseline", "fit"]).baseline_command == "fit"
    assert parser.parse_args(["smoke"]).limit == 2
    assert parser.parse_args(["benchmark-kronos"]).batch_sizes == [4, 8, 16, 32, 64]
    prediction = parser.parse_args(["predict", "--model", "kronos", "--split", "test"])
    assert prediction.seed == 42
    assert prediction.num_shards is None
    assert parser.parse_args(["evaluate"]).command == "evaluate"
    assert parser.parse_args(["report"]).command == "report"


def test_canonical_kronos_predict_rejects_limit() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["predict", "--model", "kronos", "--split", "test", "--limit", "2"]
        )
