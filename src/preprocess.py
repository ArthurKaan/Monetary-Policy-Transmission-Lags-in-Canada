"""Frequency alignment, series splicing, transformations, and stationarity tests.

Aligns raw daily/monthly/quarterly Canadian macroeconomic, dual housing (NHPI prices
and CMHC Housing Starts volume), and exogenous global control series (BoC BCPI
commodity prices & US Effective Federal Funds Rate) from data/raw/ to a clean
quarterly panel (1992Q1-latest), constructs COVID-19 impulse dummies (2020Q2-Q3),
saves both log-level and stationary transformed panels, and runs ADF & KPSS tests.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

START_QUARTER = "1992Q1"
PRE_COVID_END = "2019Q4"


def load_raw_series(name: str, raw_dir: Path = RAW_DIR) -> pd.Series:
    """Load a raw CSV as a date-indexed pandas Series."""
    df = pd.read_csv(raw_dir / f"{name}.csv", parse_dates=["date"])
    s = df.set_index("date")["value"].sort_index()
    s.name = name
    return s


def build_spliced_usdcad(legacy: pd.Series, modern: pd.Series) -> pd.Series:
    """Splice StatCan legacy USD/CAD (pre-2017) with BoC FXMUSDCAD (2017+)."""
    overlap = pd.concat([legacy.rename("legacy"), modern.rename("modern")], axis=1).dropna()
    if not overlap.empty:
        max_diff = (overlap["legacy"] - overlap["modern"]).abs().max()
        print(
            f"USD/CAD splice overlap ({overlap.index.min().strftime('%Y-%m')} to "
            f"{overlap.index.max().strftime('%Y-%m')}): max abs diff = {max_diff:.5f}"
        )

    splice_date = modern.index.min()
    spliced = pd.concat([legacy[legacy.index < splice_date], modern]).sort_index()
    spliced.name = "usdcad"
    return spliced


def build_spliced_overnight_rate(
    target_daily: pd.Series, mmf_monthly: pd.Series, bank_rate_monthly: pd.Series
) -> pd.Series:
    """Build a continuous monthly policy rate series from 1992 onward.

    Priority by period:
    1. 2009-05 onward: monthly average of BoC daily Target for the Overnight Rate (V39079).
    2. 1996-01 to 2009-04: BoC monthly Overnight Money Market Financing Rate (V122514).
    3. Prior to 1996-01: Bank Rate (V122530) minus 0.25 pp (standard operating-band spread).
    """
    target_monthly = target_daily.resample("MS").mean()
    target_monthly = target_monthly[target_monthly.index >= "2009-05-01"]

    ov_recent = pd.concat(
        [mmf_monthly.rename("mmf"), target_monthly.rename("target")], axis=1
    ).dropna()
    if not ov_recent.empty:
        mean_spread = (ov_recent["mmf"] - ov_recent["target"]).mean()
        print(
            f"Overnight rate splice (MMF vs Target, n={len(ov_recent)}m): "
            f"mean spread = {mean_spread:+.4f} pp"
        )

    bank_adj = bank_rate_monthly - 0.25
    ov_early = pd.concat(
        [bank_adj.rename("bank_minus_25bp"), mmf_monthly.rename("mmf")], axis=1
    ).loc["1996-01-01":"1996-12-01"].dropna()
    if not ov_early.empty:
        early_spread = (ov_early["bank_minus_25bp"] - ov_early["mmf"]).mean()
        print(
            f"Overnight rate splice (BankRate-25bp vs MMF in 1996): "
            f"mean spread = {early_spread:+.4f} pp"
        )

    pre_1996 = bank_adj[bank_adj.index < mmf_monthly.index.min()]
    mid_period = mmf_monthly[
        (mmf_monthly.index >= mmf_monthly.index.min())
        & (mmf_monthly.index < target_monthly.index.min())
    ]
    spliced = pd.concat([pre_1996, mid_period, target_monthly]).sort_index()
    spliced.name = "overnight_rate"
    return spliced


def to_quarterly_mean(s: pd.Series, require_full_quarter: bool = True) -> pd.Series:
    """Aggregate a monthly series to quarterly averages indexed by PeriodIndex(freq='Q')."""
    q_period = s.index.to_period("Q")
    grouped = s.groupby(q_period)
    q_mean = grouped.mean()
    if require_full_quarter:
        counts = grouped.count()
        q_mean = q_mean[counts == 3]
    return q_mean


def add_covid_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """Add pandemic lockdown/rebound impulse dummies for 2020Q2 and 2020Q3."""
    out = df.copy()
    q_str = out.index.astype(str)
    out["covid_2020q2"] = (q_str == "2020Q2").astype(float)
    out["covid_2020q3"] = (q_str == "2020Q3").astype(float)
    return out


def run_stationarity_tests(df: pd.DataFrame, sample_label: str) -> pd.DataFrame:
    """Run ADF (H0: unit root) and KPSS (H0: stationary) tests on each column."""
    rows = []
    for col in df.columns:
        series = df[col].dropna()
        adf_stat, adf_p, adf_lags, *_ = adfuller(
            series, regression="c", autolag="AIC", result_object=False
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InterpolationWarning)
            kpss_stat, kpss_p, kpss_lags, _ = kpss(
                series, regression="c", nlags="auto", result_object=False
            )

        adf_stationary = adf_p < 0.05
        kpss_stationary = kpss_p >= 0.05
        if adf_stationary and kpss_stationary:
            verdict = "Stationary"
        elif adf_stationary and not kpss_stationary:
            verdict = "Trend/Borderline (ADF yes, KPSS no)"
        elif not adf_stationary and kpss_stationary:
            verdict = "Weak/Borderline (ADF no, KPSS yes)"
        else:
            verdict = "Non-stationary"

        rows.append(
            {
                "sample": sample_label,
                "variable": col,
                "n_obs": len(series),
                "adf_stat": round(float(adf_stat), 4),
                "adf_pvalue": round(float(adf_p), 4),
                "adf_lags": int(adf_lags),
                "kpss_stat": round(float(kpss_stat), 4),
                "kpss_pvalue": round(float(kpss_p), 4),
                "kpss_lags": int(kpss_lags),
                "verdict": verdict,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load raw series
    real_gdp = load_raw_series("real_gdp")
    cpi_sa = load_raw_series("cpi_sa")
    nhpi = load_raw_series("nhpi")
    housing_starts = load_raw_series("housing_starts")
    usdcad_legacy = load_raw_series("usdcad_legacy")
    usdcad_modern = load_raw_series("usdcad_modern")
    overnight_target = load_raw_series("overnight_target")
    overnight_mmf = load_raw_series("overnight_mmf")
    bank_rate = load_raw_series("bank_rate")
    bcpi = load_raw_series("bcpi")
    us_fedfunds = load_raw_series("us_fedfunds")

    # 2. Splice monthly exchange rate and overnight policy rate
    usdcad_monthly = build_spliced_usdcad(usdcad_legacy, usdcad_modern)
    overnight_monthly = build_spliced_overnight_rate(
        overnight_target, overnight_mmf, bank_rate
    )

    # 3. Align all series to quarterly frequency
    q_gdp = real_gdp.copy()
    q_gdp.index = q_gdp.index.to_period("Q")
    q_cpi = to_quarterly_mean(cpi_sa, require_full_quarter=True)
    q_nhpi = to_quarterly_mean(nhpi, require_full_quarter=True)
    q_starts = to_quarterly_mean(housing_starts, require_full_quarter=True)
    q_rate = to_quarterly_mean(overnight_monthly, require_full_quarter=True)
    q_fx = to_quarterly_mean(usdcad_monthly, require_full_quarter=True)
    q_bcpi = to_quarterly_mean(bcpi, require_full_quarter=True)
    q_usff = to_quarterly_mean(us_fedfunds, require_full_quarter=True)

    panel = pd.concat(
        [
            q_gdp.rename("real_gdp"),
            q_cpi.rename("cpi_sa"),
            q_nhpi.rename("nhpi"),
            q_starts.rename("housing_starts"),
            q_rate.rename("overnight_rate"),
            q_fx.rename("usdcad"),
            q_bcpi.rename("bcpi"),
            q_usff.rename("us_fedfunds"),
        ],
        axis=1,
    ).dropna()

    panel = panel.loc[pd.Period(START_QUARTER, freq="Q") :]

    # 4. Construct log-levels (100 * ln for direct % interpretation)
    levels = panel.copy()
    levels["log_gdp"] = 100.0 * np.log(levels["real_gdp"])
    levels["log_cpi"] = 100.0 * np.log(levels["cpi_sa"])
    levels["log_nhpi"] = 100.0 * np.log(levels["nhpi"])
    levels["log_starts"] = 100.0 * np.log(levels["housing_starts"])
    levels["log_usdcad"] = 100.0 * np.log(levels["usdcad"])
    levels["log_bcpi"] = 100.0 * np.log(levels["bcpi"])
    levels = add_covid_dummies(levels)

    # 5. Construct stationary QoQ annualized growth rates + policy rate changes
    stationary = pd.DataFrame(index=levels.index)
    stationary["gdp_growth"] = 4.0 * levels["log_gdp"].diff()
    stationary["cpi_inflation"] = 4.0 * levels["log_cpi"].diff()
    stationary["nhpi_growth"] = 4.0 * levels["log_nhpi"].diff()
    stationary["starts_growth"] = 4.0 * levels["log_starts"].diff()
    stationary["overnight_rate"] = levels["overnight_rate"]
    stationary["d_overnight_rate"] = levels["overnight_rate"].diff()
    stationary["fx_change"] = 4.0 * levels["log_usdcad"].diff()
    stationary["bcpi_growth"] = 4.0 * levels["log_bcpi"].diff()
    stationary["us_fedfunds"] = levels["us_fedfunds"]
    stationary["d_us_fedfunds"] = levels["us_fedfunds"].diff()
    # YoY versions for exploratory visualization
    stationary["cpi_yoy"] = levels["log_cpi"].diff(4)
    stationary["nhpi_yoy"] = levels["log_nhpi"].diff(4)
    stationary["starts_yoy"] = levels["log_starts"].diff(4)
    stationary["gdp_yoy"] = levels["log_gdp"].diff(4)
    stationary = add_covid_dummies(stationary)
    stationary = stationary.dropna(
        subset=[
            "gdp_growth",
            "cpi_inflation",
            "nhpi_growth",
            "starts_growth",
            "d_overnight_rate",
            "fx_change",
            "bcpi_growth",
            "d_us_fedfunds",
        ]
    )

    # Save processed CSVs
    levels_out = levels.copy()
    levels_out.index = levels_out.index.astype(str)
    levels_out.index.name = "quarter"
    levels_out.to_csv(PROCESSED_DIR / "quarterly_levels.csv")

    stationary_out = stationary.copy()
    stationary_out.index = stationary_out.index.astype(str)
    stationary_out.index.name = "quarter"
    stationary_out.to_csv(PROCESSED_DIR / "quarterly_stationary.csv")

    # 6. Stationarity diagnostics
    level_cols = [
        "log_gdp",
        "log_cpi",
        "log_nhpi",
        "log_starts",
        "overnight_rate",
        "log_usdcad",
        "log_bcpi",
        "us_fedfunds",
    ]
    stat_cols = [
        "gdp_growth",
        "cpi_inflation",
        "nhpi_growth",
        "starts_growth",
        "overnight_rate",
        "d_overnight_rate",
        "fx_change",
        "bcpi_growth",
        "d_us_fedfunds",
    ]

    pre_end = pd.Period(PRE_COVID_END, freq="Q")
    tests = pd.concat(
        [
            run_stationarity_tests(levels[level_cols], "Full (Levels)"),
            run_stationarity_tests(stationary[stat_cols], "Full (Transformed)"),
            run_stationarity_tests(
                levels.loc[:pre_end, level_cols], "Pre-COVID 1992-2019 (Levels)"
            ),
            run_stationarity_tests(
                stationary.loc[:pre_end, stat_cols], "Pre-COVID 1992-2019 (Transformed)"
            ),
        ],
        ignore_index=True,
    )
    tests.to_csv(PROCESSED_DIR / "stationarity_tests.csv", index=False)

    print(
        f"\nQuarterly panel built: {panel.index.min()} to {panel.index.max()} "
        f"({len(panel)} quarters)"
    )
    print(f"Stationary panel rows: {len(stationary)} ({stationary.index.min()} to {stationary.index.max()})")
    print("\nStationarity Diagnostics (Full Sample):")
    full_tests = tests[tests["sample"].str.startswith("Full")]
    print(
        full_tests[
            ["sample", "variable", "adf_stat", "adf_pvalue", "kpss_stat", "kpss_pvalue", "verdict"]
        ].to_string(index=False)
    )
    print(f"\nSaved processed datasets and stationarity report to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
