from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import leonos.figures as figures
import leonos.reporting as reporting
from leonos.cli import main


def _fixture_artifacts(tmp_path: Path) -> tuple[dict[str, object], Path]:
    evaluation_root = tmp_path / "artifacts" / "evaluation" / "seed=42"
    evaluation_root.mkdir(parents=True)
    dates = pd.bdate_range("2025-01-02", periods=25)
    pd.DataFrame(
        {
            "origin": dates,
            "coverage_ok": True,
            "delta_rankic": np.linspace(-0.2, 0.2, len(dates)),
        }
    ).to_parquet(evaluation_root / "daily_metrics.parquet", index=False)

    initial_cash = 1_000_000.0
    returns = {
        "kronos": np.linspace(-0.002, 0.003, len(dates)),
        "lightgbm": np.linspace(0.001, 0.002, len(dates)),
    }
    for model, net_return in returns.items():
        wealth = initial_cash * np.cumprod(1.0 + net_return)
        path = evaluation_root / "portfolio" / model / "cost_bps=5"
        path.mkdir(parents=True)
        pd.DataFrame(
            {
                "datetime": dates,
                "net_return": net_return,
                "compounded_wealth": wealth,
                "account": wealth,
            }
        ).to_parquet(path / "qlib_report.parquet", index=False)

    reference_return = np.linspace(-0.001, 0.0015, len(dates))
    reference_wealth = initial_cash * np.cumprod(1.0 + reference_return)
    reference_path = evaluation_root / "portfolio" / "equal_weight_buy_hold" / "cost_bps=5"
    reference_path.mkdir(parents=True)
    pd.DataFrame(
        {
            "session": dates,
            "net_return": reference_return,
            "account_value": reference_wealth,
        }
    ).to_parquet(reference_path / "account.parquet", index=False)

    config: dict[str, object] = {
        "experiment": {"seed": 42},
        "paths": {"artifacts": str(tmp_path / "artifacts")},
        "portfolio": {"cost_bps_per_side": 5, "initial_cash": initial_cash},
    }
    return config, evaluation_root


def test_rankic_frame_uses_complete_twenty_session_window(tmp_path: Path) -> None:
    _, evaluation_root = _fixture_artifacts(tmp_path)

    frame = figures._rankic_frame(evaluation_root / "daily_metrics.parquet")

    assert frame["rolling_mean"].iloc[:19].isna().all()
    assert frame["rolling_mean"].iloc[19] == pytest.approx(frame["delta_rankic"].iloc[:20].mean())


def test_wealth_frame_reconciles_saved_account_convention(tmp_path: Path) -> None:
    config, evaluation_root = _fixture_artifacts(tmp_path)

    frame = figures._wealth_frame(
        evaluation_root,
        initial_cash=float(config["portfolio"]["initial_cash"]),  # type: ignore[index]
    )

    assert list(frame) == [
        "Kronos-base",
        "LightGBM",
        "95%-invested equal-dollar reference",
    ]
    assert len(frame) == 25
    assert frame.index.equals(pd.bdate_range("2025-01-02", periods=25))


def test_render_result_figures_writes_two_reproducible_pngs(tmp_path: Path) -> None:
    config, _ = _fixture_artifacts(tmp_path)
    output_dir = tmp_path / "reports" / "figures"

    paths = figures.render_result_figures(config, output_dir=output_dir)
    first_hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()
    }
    regenerated = figures.render_result_figures(config, output_dir=output_dir)
    second_hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in regenerated.items()
    }

    assert paths == {
        "rankic_difference": output_dir / "rankic-difference.png",
        "net_wealth": output_dir / "net-wealth.png",
    }
    assert first_hashes == second_hashes
    assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in paths.values())
    assert not list(output_dir.glob(".*.png"))


def test_wealth_frame_rejects_account_values_that_do_not_reconcile(tmp_path: Path) -> None:
    config, evaluation_root = _fixture_artifacts(tmp_path)
    path = evaluation_root / "portfolio" / "kronos" / "cost_bps=5" / "qlib_report.parquet"
    frame = pd.read_parquet(path)
    frame.loc[frame.index[-1], "compounded_wealth"] += 10.0
    frame.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="do not reconcile"):
        figures._wealth_frame(
            evaluation_root,
            initial_cash=float(config["portfolio"]["initial_cash"]),  # type: ignore[index]
        )


def test_report_command_returns_report_and_figure_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("experiment:\n  seed: 42\n", encoding="utf-8")
    report_path = tmp_path / "reports" / "results.md"
    figure_paths = {
        "rankic_difference": tmp_path / "rankic-difference.png",
        "net_wealth": tmp_path / "net-wealth.png",
    }
    monkeypatch.setattr(reporting, "render_results_report", lambda _config: report_path)
    monkeypatch.setattr(figures, "render_result_figures", lambda _config: figure_paths)

    assert main(["--config", str(config_path), "report"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "figures": {name: str(path) for name, path in figure_paths.items()},
        "report": str(report_path),
    }
