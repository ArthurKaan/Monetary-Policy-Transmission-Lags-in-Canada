"""Methodological triangulation: Jordà (2005) Local Projections, US Spillovers & Orderings.

Tests the flagship 7-variable Sims (1992) Small-Open-Economy SVARX (1992Q1-2026Q2
with COVID-19 impulse controls) against three independent econometric checks:

1. Jordà (2005) Local Projections (LP) with Newey-West HAC standard errors:
   Directly estimates horizon-by-horizon responses y_{t+h} - y_{t-1} to a +100 bps
   overnight rate shock under the same recursive identification controls
   (Plagborg-Møller & Wolf, 2021) without imposing multi-step VAR dynamics.

2. US Monetary Policy Spillover Block (8-Var SVARX):
   Adds the US Effective Federal Funds Rate (FRED: FEDFUNDS) ordered first in the
   global block [US Fed Funds, BCPI, GDP, CPI, Starts, NHPI, BoC Rate, FX] to
   isolate purely idiosyncratic Bank of Canada shocks from Federal Reserve moves.

3. Alternative Cholesky Ordering Sensitivity:
   - Fast Housing: [BCPI, GDP, CPI, BoC Rate, Starts, NHPI, FX]
     Allows housing starts and new home prices to react within quarter 0.

Exports comparison tables to data/processed/results/robustness/ and charts to figures/.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["svg.fonttype"] = "none"
import numpy as np
import pandas as pd
import statsmodels.api as sm

from estimate_var import (
    EXOG_COVID,
    FIGURES_DIR,
    IRF_HORIZON,
    LEVEL_COLS,
    PROCESSED_DIR,
    RESULTS_DIR,
    estimate_svar,
)

ROBUST_DIR = RESULTS_DIR / "robustness"

TARGET_VARS = ["log_gdp", "log_cpi", "log_starts", "log_nhpi"]
TARGET_TITLES = {
    "log_gdp": "Real GDP (% level)",
    "log_cpi": "Headline CPI (% level)",
    "log_starts": "Housing Starts — Activity (% level)",
    "log_nhpi": "Housing Prices — NHPI (% level)",
}


def estimate_local_projections(
    df: pd.DataFrame,
    endog_cols: list[str],
    policy_col: str,
    exog_cols: list[str],
    n_lags: int = 4,
    horizon: int = IRF_HORIZON,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Estimate recursive Jordà (2005) Local Projections with Newey-West HAC SEs."""
    data = df[endog_cols + exog_cols].dropna().copy().reset_index(drop=True)
    data["trend"] = np.arange(1, len(data) + 1, dtype=float)

    policy_pos = endog_cols.index(policy_col)
    slow_cols = endog_cols[:policy_pos]

    lag_frames = [
        data[endog_cols].shift(lag).add_suffix(f"_L{lag}")
        for lag in range(1, n_lags + 1)
    ]
    lags_df = pd.concat(lag_frames, axis=1)

    horizons = np.arange(horizon + 1)
    point = pd.DataFrame(index=horizons, columns=endog_cols, dtype=float)
    se = pd.DataFrame(index=horizons, columns=endog_cols, dtype=float)
    point.index.name = "horizon_q"
    se.index.name = "horizon_q"

    for col in endog_cols:
        for h in horizons:
            if h == 0 and col in slow_cols:
                point.loc[h, col] = 0.0
                se.loc[h, col] = 0.0
                continue
            if h == 0 and col == policy_col:
                point.loc[h, col] = 1.0
                se.loc[h, col] = 0.0
                continue

            lhs = (data[col].shift(-h) - data[col].shift(1)).rename("lhs")
            ctrl_cols = [policy_col] + slow_cols + exog_cols + ["trend"]
            rhs = pd.concat([data[ctrl_cols], lags_df], axis=1)
            rhs = sm.add_constant(rhs, has_constant="add")

            reg_df = pd.concat([lhs, rhs], axis=1).dropna()
            hac_lags = max(h + 1, 4)
            ols_res = sm.OLS(reg_df["lhs"], reg_df.drop(columns=["lhs"])).fit(
                cov_type="HAC", cov_kwds={"maxlags": hac_lags}
            )
            point.loc[h, col] = float(ols_res.params[policy_col])
            se.loc[h, col] = float(ols_res.bse[policy_col])

    lo68 = point - 1.0 * se
    hi68 = point + 1.0 * se
    lo90 = point - 1.645 * se
    hi90 = point + 1.645 * se
    return point, lo68, hi68, lo90, hi90


def plot_robustness_comparison(
    svar_baseline: dict[str, pd.DataFrame],
    lp_point: pd.DataFrame,
    lp_lo68: pd.DataFrame,
    lp_hi68: pd.DataFrame,
    alt_irfs: dict[str, pd.DataFrame],
) -> None:
    """2x2 comparison: Sims SVARX vs. Local Projections, US Fed Funds & Fast Housing."""
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), dpi=150)
    axes_flat = axes.flatten()
    h = lp_point.index.values

    for i, col in enumerate(TARGET_VARS):
        ax = axes_flat[i]
        ax.axhline(0, color="#444444", lw=0.9, ls="--")

        ax.fill_between(
            h,
            svar_baseline["lo68"][col],
            svar_baseline["hi68"][col],
            color="#2563eb",
            alpha=0.18,
            label="SVARX 68% bootstrap CI",
        )
        ax.plot(
            h,
            svar_baseline["point"][col],
            color="#1e3a8a",
            lw=2.3,
            label="7-Var Sims SVARX (Baseline)",
        )

        ax.fill_between(
            h,
            lp_lo68[col],
            lp_hi68[col],
            color="#dc2626",
            alpha=0.12,
            label="Local Proj. 68% HAC CI",
        )
        ax.plot(
            h,
            lp_point[col],
            color="#dc2626",
            lw=2.0,
            ls="-.",
            label="Jordà (2005) Local Projections",
        )

        ax.plot(
            h,
            alt_irfs["us_fedfunds_block"][col],
            color="#7c3aed",
            lw=1.8,
            ls="--",
            label="8-Var SVARX (+US Fed Funds)",
        )
        ax.plot(
            h,
            alt_irfs["alt_fast_housing"][col],
            color="#059669",
            lw=1.8,
            ls=":",
            label="Alt Ordering: Fast Housing",
        )

        ax.set_title(TARGET_TITLES[col], fontsize=11, fontweight="bold")
        ax.set_xlabel("Quarters after +100 bps hike", fontsize=9)
        ax.set_xticks(range(0, IRF_HORIZON + 1, 4))
        ax.grid(alpha=0.25)
        if i == 0:
            ax.legend(fontsize=7.6, loc="lower left", ncol=2)

    axes_flat[0].set_ylabel("% deviation from baseline")
    axes_flat[2].set_ylabel("% deviation from baseline")
    fig.suptitle(
        "Methodological Robustness: Sims SVARX vs. Jordà Local Projections, US Spillovers & Orderings\n"
        "Full Inflation-Targeting Era (1992Q1–2026Q2, COVID-19 Impulse Controls)",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIGURES_DIR / "robustness_lp_and_ordering.png", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "robustness_lp_and_ordering.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ROBUST_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    levels = pd.read_csv(PROCESSED_DIR / "quarterly_levels.csv", index_col="quarter")

    # 1. Load baseline flagship 7-variable Sims SVARX IRFs
    base_dir = RESULTS_DIR / "full_levels_varx"
    svar_baseline = {
        "point": pd.read_csv(base_dir / "irf_100bp.csv", index_col="horizon_q"),
        "lo68": pd.read_csv(base_dir / "irf_lower_68.csv", index_col="horizon_q"),
        "hi68": pd.read_csv(base_dir / "irf_upper_68.csv", index_col="horizon_q"),
    }

    # 2. Jordà (2005) Local Projections with Newey-West HAC SEs
    lp_point, lp_lo68, lp_hi68, lp_lo90, lp_hi90 = estimate_local_projections(
        levels,
        endog_cols=LEVEL_COLS,
        policy_col="overnight_rate",
        exog_cols=EXOG_COVID,
        n_lags=4,
        horizon=IRF_HORIZON,
    )
    lp_point.to_csv(ROBUST_DIR / "lp_irf_100bp.csv")
    lp_lo68.to_csv(ROBUST_DIR / "lp_lower_68.csv")
    lp_hi68.to_csv(ROBUST_DIR / "lp_upper_68.csv")
    lp_lo90.to_csv(ROBUST_DIR / "lp_lower_90.csv")
    lp_hi90.to_csv(ROBUST_DIR / "lp_upper_90.csv")

    # 3. Alternative specifications: US Fed Funds block + Fast Housing ordering
    alt_specs = {
        "us_fedfunds_block": [
            "us_fedfunds",
            "log_bcpi",
            "log_gdp",
            "log_cpi",
            "log_starts",
            "log_nhpi",
            "overnight_rate",
            "log_usdcad",
        ],
        "alt_fast_housing": [
            "log_bcpi",
            "log_gdp",
            "log_cpi",
            "overnight_rate",
            "log_starts",
            "log_nhpi",
            "log_usdcad",
        ],
    }

    alt_irfs: dict[str, pd.DataFrame] = {}
    summary_rows = []

    for label, frame in [("svarx_baseline", svar_baseline["point"]), ("jorda_lp", lp_point)]:
        for col in TARGET_VARS:
            trough_q = int(frame[col].iloc[1:].idxmin())
            summary_rows.append(
                {
                    "method": label,
                    "variable": col,
                    "trough_quarter": trough_q,
                    "trough_response_per_100bp": round(float(frame.loc[trough_q, col]), 4),
                    "q4_response": round(float(frame.loc[4, col]), 4),
                    "q8_response": round(float(frame.loc[8, col]), 4),
                    "q12_response": round(float(frame.loc[12, col]), 4),
                }
            )

    for alt_name, cols_order in alt_specs.items():
        bundle = estimate_svar(
            levels,
            cols=cols_order,
            policy_col="overnight_rate",
            name=alt_name,
            sample_label="Full Era (1992Q1–2026Q2)",
            spec_type=f"Robustness: {alt_name}",
            exog_cols=EXOG_COVID,
            trend="ct",
            forced_lag=4,
            n_bootstrap=200,
        )
        alt_irfs[alt_name] = bundle.irf_100bp
        bundle.irf_100bp.to_csv(ROBUST_DIR / f"{alt_name}_irf_100bp.csv")
        for col in TARGET_VARS:
            trough_q = int(bundle.irf_100bp[col].iloc[1:].idxmin())
            summary_rows.append(
                {
                    "method": alt_name,
                    "variable": col,
                    "trough_quarter": trough_q,
                    "trough_response_per_100bp": round(float(bundle.irf_100bp.loc[trough_q, col]), 4),
                    "q4_response": round(float(bundle.irf_100bp.loc[4, col]), 4),
                    "q8_response": round(float(bundle.irf_100bp.loc[8, col]), 4),
                    "q12_response": round(float(bundle.irf_100bp.loc[12, col]), 4),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(ROBUST_DIR / "robustness_summary.csv", index=False)

    plot_robustness_comparison(svar_baseline, lp_point, lp_lo68, lp_hi68, alt_irfs)

    print("\n=== Methodological Robustness Summary (Full Era 1992Q1-2026Q2) ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved robustness outputs to {ROBUST_DIR} and figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
