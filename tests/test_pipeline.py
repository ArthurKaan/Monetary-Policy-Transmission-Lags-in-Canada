"""Unit and econometric invariant tests for the Canadian Monetary Policy SVAR/VARX pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import estimate_var as ev  # noqa: E402
import preprocess as pp  # noqa: E402
import robustness as rb  # noqa: E402


def _make_synthetic_panel(n_quarters: int = 120, seed: int = 123) -> pd.DataFrame:
    """Generate a stable synthetic 6-var + 4-exog quarterly panel for fast testing."""
    rng = np.random.default_rng(seed)
    idx = pd.period_range("1995Q1", periods=n_quarters, freq="Q")
    t = np.arange(n_quarters, dtype=float)

    # Stationary AR(1) innovations + gentle linear trends
    def ar1(phi: float, scale: float) -> np.ndarray:
        x = np.zeros(n_quarters)
        eps = rng.normal(0.0, scale, size=n_quarters)
        for i in range(1, n_quarters):
            x[i] = phi * x[i - 1] + eps[i]
        return x

    rate = 3.0 + ar1(0.75, 0.4)
    gdp = 1400.0 + 0.5 * t + ar1(0.6, 0.8) - 0.15 * np.roll(rate, 2)
    cpi = 460.0 + 0.45 * t + ar1(0.65, 0.35) - 0.08 * np.roll(rate, 3)
    starts = 520.0 + 0.1 * t + ar1(0.6, 2.5) - 0.6 * np.roll(rate, 1)
    nhpi = 450.0 + 0.6 * t + ar1(0.7, 0.5) - 0.2 * np.roll(rate, 2)
    fx = 25.0 + ar1(0.7, 1.5) - 0.25 * rate

    df = pd.DataFrame(
        {
            "log_gdp": gdp,
            "log_cpi": cpi,
            "log_starts": starts,
            "log_nhpi": nhpi,
            "overnight_rate": rate,
            "log_usdcad": fx,
            "log_bcpi": 550.0 + ar1(0.7, 4.0),
            "us_fedfunds": 2.5 + ar1(0.75, 0.35),
        },
        index=idx,
    )
    df = pp.add_covid_dummies(df)
    df.index = df.index.astype(str)
    df.index.name = "quarter"
    return df


def test_usdcad_splice_continuity() -> None:
    legacy = pd.Series(
        [1.30, 1.32, 1.34, 1.35],
        index=pd.to_datetime(["2016-11-01", "2016-12-01", "2017-01-01", "2017-02-01"]),
    )
    modern = pd.Series(
        [1.3401, 1.3502, 1.36],
        index=pd.to_datetime(["2017-01-01", "2017-02-01", "2017-03-01"]),
    )
    spliced = pp.build_spliced_usdcad(legacy, modern)
    assert len(spliced) == 5
    assert spliced.index.is_monotonic_increasing
    assert not spliced.index.duplicated().any()
    assert spliced.loc[pd.Timestamp("2016-12-01")] == pytest.approx(1.32)
    assert spliced.loc[pd.Timestamp("2017-01-01")] == pytest.approx(1.3401)


def test_overnight_rate_splice_priority() -> None:
    bank_rate = pd.Series(
        [5.25, 5.00, 4.75],
        index=pd.to_datetime(["1995-12-01", "1996-01-01", "2009-05-01"]),
    )
    mmf = pd.Series(
        [4.70, 0.26],
        index=pd.to_datetime(["1996-01-01", "2009-05-01"]),
    )
    target_daily = pd.Series(
        [0.25, 0.25],
        index=pd.to_datetime(["2009-05-04", "2009-05-15"]),
    )
    spliced = pp.build_spliced_overnight_rate(target_daily, mmf, bank_rate)
    assert spliced.loc[pd.Timestamp("1995-12-01")] == pytest.approx(5.00)
    assert spliced.loc[pd.Timestamp("1996-01-01")] == pytest.approx(4.70)
    assert spliced.loc[pd.Timestamp("2009-05-01")] == pytest.approx(0.25)


def test_to_quarterly_mean_drops_incomplete_quarters() -> None:
    s = pd.Series(
        [1.0, 2.0, 3.0, 10.0, 20.0],
        index=pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01"]),
    )
    q = pp.to_quarterly_mean(s, require_full_quarter=True)
    assert list(q.index.astype(str)) == ["2024Q1"]
    assert q.iloc[0] == pytest.approx(2.0)


def test_covid_dummies_exact_quarters() -> None:
    idx = pd.period_range("2020Q1", periods=4, freq="Q")
    df = pp.add_covid_dummies(pd.DataFrame({"x": [1, 2, 3, 4]}, index=idx))
    assert df["covid_2020q2"].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert df["covid_2020q3"].tolist() == [0.0, 0.0, 1.0, 0.0]


def test_svar_cholesky_identification_and_invariants() -> None:
    df = _make_synthetic_panel()
    bundle = ev.estimate_svar(
        df,
        cols=ev.LEVEL_COLS,
        policy_col="overnight_rate",
        name="synthetic_test",
        sample_label="Synthetic",
        spec_type="Test VARX",
        exog_cols=ev.EXOG_COVID,
        trend="ct",
        forced_lag=2,
        n_bootstrap=40,
    )

    # Exact +100 bps (+1.0 pp) normalization on impact
    assert bundle.irf_100bp.loc[0, "overnight_rate"] == pytest.approx(1.0, abs=1e-10)

    # Cholesky recursive zero restrictions for slow-moving variables at h=0
    for slow_col in ["log_bcpi", "log_gdp", "log_cpi", "log_starts", "log_nhpi"]:
        assert bundle.irf_100bp.loc[0, slow_col] == pytest.approx(0.0, abs=1e-10)

    # Confidence band ordering across all horizons & variables
    assert (bundle.irf_lower_90 <= bundle.irf_lower_68 + 1e-10).all().all()
    assert (bundle.irf_lower_68 <= bundle.irf_upper_68 + 1e-10).all().all()
    assert (bundle.irf_upper_68 <= bundle.irf_upper_90 + 1e-10).all().all()

    # FEVD shares bounded in [0, 100]
    assert (bundle.fevd_policy >= 0.0).all().all()
    assert (bundle.fevd_policy <= 100.0).all().all()

    # Companion stability matches eigenvalue modulus < 1
    assert bundle.diagnostics["is_stable"] == (
        bundle.diagnostics["max_companion_eigenvalue"] < 1.0
    )


def test_local_projections_recursive_restrictions_and_bands() -> None:
    df = _make_synthetic_panel()
    point, lo68, hi68, lo90, hi90 = rb.estimate_local_projections(
        df,
        endog_cols=ev.LEVEL_COLS,
        policy_col="overnight_rate",
        exog_cols=ev.EXOG_COVID,
        n_lags=2,
        horizon=8,
    )
    assert point.loc[0, "overnight_rate"] == pytest.approx(1.0)
    for slow_col in ["log_bcpi", "log_gdp", "log_cpi", "log_starts", "log_nhpi"]:
        assert point.loc[0, slow_col] == pytest.approx(0.0)

    assert (lo90 <= lo68 + 1e-10).all().all()
    assert (lo68 <= point + 1e-10).all().all()
    assert (point <= hi68 + 1e-10).all().all()
    assert (hi68 <= hi90 + 1e-10).all().all()
