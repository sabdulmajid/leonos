from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leonos.scenario import paired_block_bootstrap_paths


def _returns(kronos: list[float], lightgbm: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session": pd.bdate_range("2025-01-02", periods=len(kronos)),
            "kronos": kronos,
            "lightgbm": lightgbm,
            "reference": np.zeros(len(kronos)),
        }
    )


def _rankic(kronos: list[float], lightgbm: list[float]) -> pd.DataFrame:
    first = np.asarray(kronos, dtype=float)
    second = np.asarray(lightgbm, dtype=float)
    return pd.DataFrame(
        {
            "origin": pd.bdate_range("2025-01-02", periods=len(first)),
            "kronos_rankic": first,
            "lightgbm_rankic": second,
            "delta_rankic": first - second,
        }
    )


def test_paired_block_bootstrap_is_deterministic_and_preserves_pairing() -> None:
    frame = _returns([0.01, -0.02, 0.03, 0.01], [0.01, -0.02, 0.03, 0.01])
    first, first_summary = paired_block_bootstrap_paths(
        {(42, 5): frame},
        {42: _rankic([0.2, -0.1, 0.3, 0.0], [0.1, -0.2, 0.1, 0.0])},
        replicates=80,
        block_length=2,
        seed=9,
        batch_size=13,
    )
    second, second_summary = paired_block_bootstrap_paths(
        {(42, 5): frame},
        {42: _rankic([0.2, -0.1, 0.3, 0.0], [0.1, -0.2, 0.1, 0.0])},
        replicates=80,
        block_length=2,
        seed=9,
        batch_size=13,
    )
    pd.testing.assert_frame_equal(first, second)
    assert first_summary == second_summary
    np.testing.assert_allclose(
        first["kronos_ending_value"], first["lightgbm_ending_value"]
    )
    assert (
        first_summary["scenarios"]["seed=42/cost_bps=5"]["paired_comparison"][
            "median_kronos_minus_lightgbm"
        ]
        == pytest.approx(0.0)
    )
    assert first_summary["scenarios"]["seed=42/cost_bps=5"]["rankic"][
        "observed_delta_kronos_minus_lightgbm"
    ] == pytest.approx(0.1)


def test_shared_draws_make_ordered_model_comparison_exact() -> None:
    frame = _returns([0.02] * 8, [0.01] * 8)
    distribution, summary = paired_block_bootstrap_paths(
        {(42, 0): frame, (43, 0): frame},
        {
            42: _rankic([0.1] * 8, [0.0] * 8),
            43: _rankic([0.1] * 8, [0.0] * 8),
        },
        replicates=50,
        block_length=3,
        seed=11,
    )
    assert len(distribution) == 100
    assert (distribution["kronos_ending_value"] > distribution["lightgbm_ending_value"]).all()
    assert (distribution["reference_ending_value"] == 100.0).all()
    for result in summary["scenarios"].values():
        assert (
            result["paired_comparison"][
                "resampled_fraction_kronos_beats_lightgbm"
            ]
            == 1.0
        )
        assert (
            result["models"]["kronos"][
                "resampled_fraction_ending_below_initial"
            ]
            == 0.0
        )
        assert result["rankic"]["resampled_delta_95_interval"] == pytest.approx(
            [0.1, 0.1]
        )


@pytest.mark.parametrize(
    "frame,message",
    [
        (
            pd.DataFrame(
                {
                    "session": ["2025-01-02"],
                    "kronos": [0.0],
                    "lightgbm": [0.0],
                }
            ),
            "missing columns",
        ),
        (
            _returns([0.0, -1.0], [0.0, 0.0]),
            "greater than -1",
        ),
    ],
)
def test_invalid_return_panels_fail_closed(frame: pd.DataFrame, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        paired_block_bootstrap_paths(
            {(42, 5): frame},
            {42: _rankic([0.0] * len(frame), [0.0] * len(frame))},
            replicates=10,
            block_length=2,
            seed=1,
        )
