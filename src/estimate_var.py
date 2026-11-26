"""5-Variable Open-Economy SVAR estimation, diagnostics, IRFs, and FEVD.

Estimates Cholesky-identified Structural VARs for Canada over:
1. Full inflation-targeting era (1992Q1-latest)
2. Pre-COVID robustness sample (1992Q1-2019Q4)

Both complementary specifications are estimated:
- Log-Levels SVAR (Sims-Stock-Watson / CEE specification):
    [log_gdp, log_cpi, log_nhpi, overnight_rate, log_usdcad]
  Directly yields cumulative % deviations in real output, headline price level,
  and new housing prices following a +100 bps (+1 pp) unexpected overnight rate hike.
- Stationary Growth-Rate SVAR:
    [gdp_growth, cpi_inflation, nhpi_growth, d_overnight_rate, fx_change]
  Ensures strict I(0) covariance stationarity and measures QoQ annualized rate
  responses to a +100 bps policy tightening shock.

Outputs tables and JSON summaries to data/processed/results/ and publication-ready
charts to figures/.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["svg.fonttype"] = "none"
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
RESULTS_DIR = PROCESSED_DIR / "results"
FIGURES_DIR = ROOT_DIR / "figures"

PRE_COVID_END = "2019Q4"
IRF_HORIZON = 20  # 20 quarters = 5 years
FEVD_HORIZONS = [1, 4, 8, 12, 16, 20]
BOOTSTRAP_REPS = 1000
RANDOM_SEED = 42

LEVEL_COLS = ["log_gdp", "log_cpi", "log_nhpi", "overnight_rate", "log_usdcad"]
STAT_COLS = ["gdp_growth", "cpi_inflation", "nhpi_growth", "d_overnight_rate", "fx_change"]

LABELS = {
    "log_gdp": "Real GDP (% level)",
    "log_cpi": "Headline CPI (% level)",
    "log_nhpi": "Housing Prices — NHPI (% level)",
    "overnight_rate": "Overnight Rate (pp)",
    "log_usdcad": "USD/CAD FX (% level, + = CAD weaker)",
    "gdp_growth": "Real GDP Growth (QoQ ann. %)",
    "cpi_inflation": "Headline CPI Inflation (QoQ ann. %)",
    "nhpi_growth": "Housing Price Growth — NHPI (QoQ ann. %)",
    "d_overnight_rate": "Overnight Rate Change (pp)",
    "fx_change": "USD/CAD Change (QoQ ann. %)",
}


@dataclass
class SVARBundle:
    name: str
    sample_label: str
    spec_type: str
    policy_col: str
    n_obs: int
    start_quarter: str
    end_quarter: str
    selected_lag: int
    lag_table: pd.DataFrame
    diagnostics: dict[str, Any]
    shock_1sd_pp: float
    irf_100bp: pd.DataFrame
    irf_lower_68: pd.DataFrame
    irf_upper_68: pd.DataFrame
    irf_lower_90: pd.DataFrame
    irf_upper_90: pd.DataFrame
    fevd_policy: pd.DataFrame
    peak_summary: pd.DataFrame


def select_optimal_lag(
    df: pd.DataFrame, maxlags: int = 6, trend: str = "c"
) -> tuple[int, pd.DataFrame]:
    """Evaluate AIC, BIC, HQIC, and FPE up to maxlags and return recommended lag."""
    model = VAR(df)
    rows = []
    for p in range(1, maxlags + 1):
        res = model.fit(p, trend=trend)
        rows.append(
            {
                "lag": p,
                "aic": float(res.aic),
                "bic": float(res.bic),
                "hqic": float(res.hqic),
                "fpe": float(res.fpe),
            }
        )
    lag_df = pd.DataFrame(rows)
    aic_lag = int(lag_df.loc[lag_df["aic"].idxmin(), "lag"])
    hqic_lag = int(lag_df.loc[lag_df["hqic"].idxmin(), "lag"])
    bic_lag = int(lag_df.loc[lag_df["bic"].idxmin(), "lag"])
    fpe_lag = int(lag_df.loc[lag_df["fpe"].idxmin(), "lag"])

    # In quarterly monetary VARs, FPE/AIC within 1..6 lags balances autocorrelation
    # removal against parameter proliferation (5 equations * 5 vars * p lags).
    chosen = max(2, fpe_lag)
    lag_df["selected_aic"] = lag_df["lag"] == aic_lag
    lag_df["selected_bic"] = lag_df["lag"] == bic_lag
    lag_df["selected_hqic"] = lag_df["lag"] == hqic_lag
    lag_df["selected_fpe"] = lag_df["lag"] == fpe_lag
    return chosen, lag_df


def residual_bootstrap_irfs(
    res: Any,
    policy_idx: int,
    horizon: int,
    trend: str = "c",
    n_reps: int = BOOTSTRAP_REPS,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Nonparametric residual bootstrap for +100 bps normalized orthogonalized IRFs.

    Returns:
        point_irf: shape (horizon + 1, k) normalized to +1.0 pp policy shock on impact.
        boot_irfs: shape (n_valid_reps, horizon + 1, k)
    """
    rng = np.random.default_rng(seed)
    k = res.neqs
    p = res.k_ar
    y = np.asarray(res.endog, dtype=float)
    n_total = len(y)
    resid = np.asarray(res.resid, dtype=float)
    resid_centered = resid - resid.mean(axis=0, keepdims=True)
    n_eff = len(resid)

    # Point estimate normalized to +100 bps (+1.0 pp) shock to policy variable at h=0
    point_orth = res.irf(horizon).orth_irfs[:, :, policy_idx]
    impact_1sd = float(point_orth[0, policy_idx])
    point_norm = point_orth / impact_1sd

    # Deterministic terms: intercept (+ optional linear trend from params)
    coefs = np.asarray(res.coefs, dtype=float)  # shape (p, k, k)
    intercept = np.asarray(res.intercept, dtype=float)
    trend_vec = (
        np.asarray(res.params.loc["trend"], dtype=float)
        if trend == "ct" and "trend" in res.params.index
        else np.zeros(k, dtype=float)
    )

    boot_list = []
    for _ in range(n_reps):
        idx = rng.integers(0, n_eff, size=n_eff)
        u_star = resid_centered[idx]
        y_star = np.zeros((n_total, k), dtype=float)
        y_star[:p] = y[:p]
        for t in range(p, n_total):
            # statsmodels trend index starts at 1 for the first effective obs (t = p)
            trend_t = (t - p + 1) if trend == "ct" else 0
            pred = intercept + trend_vec * trend_t
            for lag_i in range(p):
                pred += coefs[lag_i] @ y_star[t - 1 - lag_i]
            y_star[t] = pred + u_star[t - p]

        try:
            boot_res = VAR(y_star).fit(p, trend=trend)
            b_orth = boot_res.irf(horizon).orth_irfs[:, :, policy_idx]
            b_impact = float(b_orth[0, policy_idx])
            if abs(b_impact) > 1e-8 and np.all(np.isfinite(b_orth)):
                boot_list.append(b_orth / b_impact)
        except Exception:
            continue

    boot_arr = np.asarray(boot_list, dtype=float)
    return point_norm, boot_arr


def estimate_svar(
    df: pd.DataFrame,
    cols: list[str],
    policy_col: str,
    name: str,
    sample_label: str,
    spec_type: str,
    trend: str = "c",
    forced_lag: int | None = None,
) -> SVARBundle:
    """Fit a Cholesky-identified SVAR, run diagnostics, bootstrap IRFs, and compute FEVD."""
    sub = df[cols].dropna().copy()
    sub.index = pd.PeriodIndex(sub.index, freq="Q")
    chosen_lag, lag_table = select_optimal_lag(sub, maxlags=6, trend=trend)
    lag = forced_lag if forced_lag is not None else chosen_lag

    model = VAR(sub)
    res = model.fit(lag, trend=trend)

    # Stability (companion matrix eigenvalues)
    roots = np.asarray(res.roots)
    inv_root_moduli = 1.0 / np.abs(roots)
    max_eigenvalue = float(np.max(inv_root_moduli))
    is_stable = bool(res.is_stable())

    # Serial correlation (Portmanteau whiteness test) & Normality (Jarque-Bera)
    whiteness_nlags = max(lag + 4, 12)
    whiteness = res.test_whiteness(nlags=whiteness_nlags)
    normality = res.test_normality()

    diagnostics = {
        "n_obs_effective": int(res.nobs),
        "lag_order": int(lag),
        "trend": trend,
        "max_companion_eigenvalue": round(max_eigenvalue, 4),
        "is_stable": is_stable,
        "whiteness_nlags": int(whiteness_nlags),
        "whiteness_stat": round(float(whiteness.test_statistic), 4),
        "whiteness_pvalue": round(float(whiteness.pvalue), 4),
        "normality_jb_stat": round(float(normality.test_statistic), 4),
        "normality_jb_pvalue": round(float(normality.pvalue), 4),
        "aic": round(float(res.aic), 4),
        "bic": round(float(res.bic), 4),
        "hqic": round(float(res.hqic), 4),
    }

    policy_idx = cols.index(policy_col)
    irf_obj = res.irf(IRF_HORIZON)
    shock_1sd_pp = float(irf_obj.orth_irfs[0, policy_idx, policy_idx])

    point_norm, boot_arr = residual_bootstrap_irfs(
        res, policy_idx=policy_idx, horizon=IRF_HORIZON, trend=trend
    )

    horizons = np.arange(IRF_HORIZON + 1)
    irf_100bp = pd.DataFrame(point_norm, index=horizons, columns=cols)
    irf_100bp.index.name = "horizon_q"

    lo68 = pd.DataFrame(np.percentile(boot_arr, 16, axis=0), index=horizons, columns=cols)
    hi68 = pd.DataFrame(np.percentile(boot_arr, 84, axis=0), index=horizons, columns=cols)
    lo90 = pd.DataFrame(np.percentile(boot_arr, 5, axis=0), index=horizons, columns=cols)
    hi90 = pd.DataFrame(np.percentile(boot_arr, 95, axis=0), index=horizons, columns=cols)
    for frame in (lo68, hi68, lo90, hi90):
        frame.index.name = "horizon_q"

    # FEVD: share of each variable's forecast error variance explained by monetary policy shock
    fevd_obj = res.fevd(max(FEVD_HORIZONS))
    # fevd_obj.decomp has shape (k, horizon, k): [response_var, h_idx, shock_var]
    fevd_rows = []
    for h in FEVD_HORIZONS:
        row: dict[str, Any] = {"horizon_q": h}
        for i, col in enumerate(cols):
            row[col] = round(100.0 * float(fevd_obj.decomp[i, h - 1, policy_idx]), 2)
        fevd_rows.append(row)
    fevd_policy = pd.DataFrame(fevd_rows).set_index("horizon_q")

    # Peak transmission lag summary (for non-policy macro targets: GDP, CPI, Housing, FX)
    peak_rows = []
    for col in cols:
        if col == policy_col:
            continue
        series = irf_100bp[col]
        # For macro contractionary responses (GDP, CPI, Housing), track peak negative trough
        # as well as peak absolute response
        min_h = int(series.iloc[1:].idxmin())  # exclude h=0 (zero by Cholesky for slow vars)
        min_val = float(series.loc[min_h])
        abs_h = int(series.iloc[1:].abs().idxmax())
        abs_val = float(series.loc[abs_h])
        sig_68_at_trough = bool(hi68.loc[min_h, col] < 0)
        sig_90_at_trough = bool(hi90.loc[min_h, col] < 0)
        peak_rows.append(
            {
                "model": name,
                "variable": col,
                "trough_quarter": min_h,
                "trough_response_per_100bp": round(min_val, 4),
                "sig_68_negative": sig_68_at_trough,
                "sig_90_negative": sig_90_at_trough,
                "peak_abs_quarter": abs_h,
                "peak_abs_response": round(abs_val, 4),
                "fevd_q4_pct": float(fevd_policy.loc[4, col]),
                "fevd_q8_pct": float(fevd_policy.loc[8, col]),
                "fevd_q12_pct": float(fevd_policy.loc[12, col]),
                "fevd_q20_pct": float(fevd_policy.loc[20, col]),
            }
        )
    peak_summary = pd.DataFrame(peak_rows)

    return SVARBundle(
        name=name,
        sample_label=sample_label,
        spec_type=spec_type,
        policy_col=policy_col,
        n_obs=len(sub),
        start_quarter=str(sub.index.min()),
        end_quarter=str(sub.index.max()),
        selected_lag=lag,
        lag_table=lag_table,
        diagnostics=diagnostics,
        shock_1sd_pp=round(shock_1sd_pp, 4),
        irf_100bp=irf_100bp,
        irf_lower_68=lo68,
        irf_upper_68=hi68,
        irf_lower_90=lo90,
        irf_upper_90=hi90,
        fevd_policy=fevd_policy,
        peak_summary=peak_summary,
    )


def save_bundle(bundle: SVARBundle) -> None:
    """Persist CSVs and JSON diagnostics for an estimated SVAR bundle."""
    out_dir = RESULTS_DIR / bundle.name
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle.lag_table.to_csv(out_dir / "lag_selection.csv", index=False)
    bundle.irf_100bp.to_csv(out_dir / "irf_100bp.csv")
    bundle.irf_lower_68.to_csv(out_dir / "irf_lower_68.csv")
    bundle.irf_upper_68.to_csv(out_dir / "irf_upper_68.csv")
    bundle.irf_lower_90.to_csv(out_dir / "irf_lower_90.csv")
    bundle.irf_upper_90.to_csv(out_dir / "irf_upper_90.csv")
    bundle.fevd_policy.to_csv(out_dir / "fevd_policy.csv")
    bundle.peak_summary.to_csv(out_dir / "peak_summary.csv", index=False)

    meta = {
        "name": bundle.name,
        "sample_label": bundle.sample_label,
        "spec_type": bundle.spec_type,
        "policy_col": bundle.policy_col,
        "n_obs": bundle.n_obs,
        "start_quarter": bundle.start_quarter,
        "end_quarter": bundle.end_quarter,
        "selected_lag": bundle.selected_lag,
        "shock_1sd_pp": bundle.shock_1sd_pp,
        "diagnostics": bundle.diagnostics,
    }
    (out_dir / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def plot_irf_grid(bundle: SVARBundle, filename: str) -> None:
    """Plot +100 bps monetary policy shock IRFs with 68% and 90% bootstrap bands."""
    cols = list(bundle.irf_100bp.columns)
    horizons = bundle.irf_100bp.index.values

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=150)
    axes_flat = axes.flatten()

    for i, col in enumerate(cols):
        ax = axes_flat[i]
        ax.axhline(0, color="#444444", linewidth=0.9, linestyle="--")
        ax.fill_between(
            horizons,
            bundle.irf_lower_90[col],
            bundle.irf_upper_90[col],
            color="#3b82f6",
            alpha=0.18,
            label="90% bootstrap CI",
        )
        ax.fill_between(
            horizons,
            bundle.irf_lower_68[col],
            bundle.irf_upper_68[col],
            color="#2563eb",
            alpha=0.32,
            label="68% bootstrap CI",
        )
        ax.plot(horizons, bundle.irf_100bp[col], color="#1e3a8a", linewidth=2.2, label="Point IRF")

        # Mark trough quarter for macro response variables
        if col != bundle.policy_col and col not in ("log_usdcad", "fx_change"):
            trough_q = int(bundle.irf_100bp[col].iloc[1:].idxmin())
            trough_val = float(bundle.irf_100bp.loc[trough_q, col])
            ax.scatter([trough_q], [trough_val], color="#dc2626", zorder=6, s=46)
            x_offset = -5.5 if trough_q >= 15 else 2.2
            y_span = float(bundle.irf_upper_90[col].max() - bundle.irf_lower_90[col].min())
            y_offset = -0.14 * y_span if trough_val > bundle.irf_lower_90[col].min() + 0.25 * y_span else 0.14 * y_span
            ax.annotate(
                f"Trough: Q{trough_q} ({trough_val:+.2f})",
                xy=(trough_q, trough_val),
                xytext=(max(0.5, min(trough_q + x_offset, 13.0)), trough_val + y_offset),
                fontsize=8.5,
                color="#991b1b",
                fontweight="bold",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#fca5a5", "alpha": 0.9},
                arrowprops={"arrowstyle": "->", "color": "#dc2626", "lw": 1.1},
                zorder=7,
            )

        ax.set_title(LABELS.get(col, col), fontsize=11, fontweight="bold")
        ax.set_xlabel("Quarters after +100 bps shock", fontsize=9)
        ax.set_xticks(range(0, IRF_HORIZON + 1, 4))
        ax.grid(alpha=0.25)
        if i == 0:
            ax.legend(fontsize=8, loc="best")

    # Hide the unused 6th panel and place concise model metadata there
    axes_flat[5].axis("off")
    info_text = (
        f"Specification: {bundle.spec_type}\n"
        f"Sample: {bundle.start_quarter}–{bundle.end_quarter} (T = {bundle.n_obs})\n"
        f"VAR lag order: p = {bundle.selected_lag}\n"
        f"1 s.d. policy shock: {bundle.shock_1sd_pp * 100:.1f} bps\n"
        f"Max eigenvalue: {bundle.diagnostics['max_companion_eigenvalue']:.3f}\n"
        f"Whiteness p-value: {bundle.diagnostics['whiteness_pvalue']:.3f}\n"
        f"Bootstrap reps: {BOOTSTRAP_REPS}"
    )
    axes_flat[5].text(
        0.05,
        0.5,
        info_text,
        va="center",
        fontsize=10,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1"},
    )

    fig.suptitle(
        f"Impulse Responses to a +100 bps Bank of Canada Overnight Rate Hike\n"
        f"{bundle.sample_label} — {bundle.spec_type}",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIGURES_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def plot_housing_vs_inflation_comparison(
    full_levels: SVARBundle, precovid_levels: SVARBundle
) -> None:
    """Side-by-side comparison of Housing (NHPI) vs Headline CPI IRFs and FEVD."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=150)
    horizons = full_levels.irf_100bp.index.values

    # Panel 1: Full sample Housing vs CPI
    ax = axes[0]
    ax.axhline(0, color="#444444", linewidth=0.9, linestyle="--")
    ax.plot(horizons, full_levels.irf_100bp["log_nhpi"], color="#dc2626", lw=2.3, label="Housing (NHPI)")
    ax.fill_between(
        horizons,
        full_levels.irf_lower_68["log_nhpi"],
        full_levels.irf_upper_68["log_nhpi"],
        color="#dc2626",
        alpha=0.16,
    )
    ax.plot(horizons, full_levels.irf_100bp["log_cpi"], color="#1d4ed8", lw=2.3, label="Headline CPI")
    ax.fill_between(
        horizons,
        full_levels.irf_lower_68["log_cpi"],
        full_levels.irf_upper_68["log_cpi"],
        color="#1d4ed8",
        alpha=0.16,
    )
    ax.set_title("Full Sample (1992–2026)\n% Response to +100 bps Hike", fontsize=11, fontweight="bold")
    ax.set_xlabel("Quarters after shock")
    ax.set_ylabel("% deviation from baseline")
    ax.set_xticks(range(0, IRF_HORIZON + 1, 4))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)

    # Panel 2: Pre-COVID sample Housing vs CPI
    ax = axes[1]
    ax.axhline(0, color="#444444", linewidth=0.9, linestyle="--")
    ax.plot(horizons, precovid_levels.irf_100bp["log_nhpi"], color="#dc2626", lw=2.3, label="Housing (NHPI)")
    ax.fill_between(
        horizons,
        precovid_levels.irf_lower_68["log_nhpi"],
        precovid_levels.irf_upper_68["log_nhpi"],
        color="#dc2626",
        alpha=0.16,
    )
    ax.plot(horizons, precovid_levels.irf_100bp["log_cpi"], color="#1d4ed8", lw=2.3, label="Headline CPI")
    ax.fill_between(
        horizons,
        precovid_levels.irf_lower_68["log_cpi"],
        precovid_levels.irf_upper_68["log_cpi"],
        color="#1d4ed8",
        alpha=0.16,
    )
    ax.set_title("Pre-COVID Robustness (1992–2019)\n% Response to +100 bps Hike", fontsize=11, fontweight="bold")
    ax.set_xlabel("Quarters after shock")
    ax.set_xticks(range(0, IRF_HORIZON + 1, 4))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)

    # Panel 3: FEVD share driven by monetary policy shock (Pre-COVID vs Full at Q8, Q12, Q20)
    ax = axes[2]
    fevd_h = [4, 8, 12, 20]
    x = np.arange(len(fevd_h))
    width = 0.25
    ax.bar(
        x - width,
        [precovid_levels.fevd_policy.loc[h, "log_gdp"] for h in fevd_h],
        width=width,
        color="#059669",
        label="Real GDP",
    )
    ax.bar(
        x,
        [precovid_levels.fevd_policy.loc[h, "log_nhpi"] for h in fevd_h],
        width=width,
        color="#dc2626",
        label="Housing (NHPI)",
    )
    ax.bar(
        x + width,
        [precovid_levels.fevd_policy.loc[h, "log_cpi"] for h in fevd_h],
        width=width,
        color="#1d4ed8",
        label="Headline CPI",
    )
    ax.set_title("FEVD: Variance Driven by Policy Shocks\n(Pre-COVID 1992–2019 Log-Levels SVAR)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{h}" for h in fevd_h])
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Share of variance (%)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "housing_vs_inflation_summary.png", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "housing_vs_inflation_summary.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    levels = pd.read_csv(PROCESSED_DIR / "quarterly_levels.csv", index_col="quarter")
    stationary = pd.read_csv(PROCESSED_DIR / "quarterly_stationary.csv", index_col="quarter")

    levels_precovid = levels.loc[:PRE_COVID_END]
    stationary_precovid = stationary.loc[:PRE_COVID_END]

    bundles = [
        estimate_svar(
            levels,
            LEVEL_COLS,
            policy_col="overnight_rate",
            name="full_levels",
            sample_label="Full Era (1992Q1–Latest)",
            spec_type="Log-Levels 5-Var Open-Economy SVAR (trend='ct')",
            trend="ct",
        ),
        estimate_svar(
            levels_precovid,
            LEVEL_COLS,
            policy_col="overnight_rate",
            name="precovid_levels",
            sample_label="Pre-COVID Era (1992Q1–2019Q4)",
            spec_type="Log-Levels 5-Var Open-Economy SVAR (trend='ct')",
            trend="ct",
        ),
        estimate_svar(
            stationary,
            STAT_COLS,
            policy_col="d_overnight_rate",
            name="full_stationary",
            sample_label="Full Era (1992Q2–Latest)",
            spec_type="Stationary Growth-Rate 5-Var SVAR",
            trend="c",
        ),
        estimate_svar(
            stationary_precovid,
            STAT_COLS,
            policy_col="d_overnight_rate",
            name="precovid_stationary",
            sample_label="Pre-COVID Era (1992Q2–2019Q4)",
            spec_type="Stationary Growth-Rate 5-Var SVAR",
            trend="c",
        ),
    ]

    all_peaks = []
    for bundle in bundles:
        save_bundle(bundle)
        plot_irf_grid(bundle, f"irf_{bundle.name}.png")
        all_peaks.append(bundle.peak_summary)

        print(f"\n=== {bundle.name} ({bundle.start_quarter} to {bundle.end_quarter}, T={bundle.n_obs}, p={bundle.selected_lag}) ===")
        print(
            f"Stable: {bundle.diagnostics['is_stable']} "
            f"(max eig={bundle.diagnostics['max_companion_eigenvalue']:.3f}) | "
            f"Whiteness p={bundle.diagnostics['whiteness_pvalue']:.3f} | "
            f"1 s.d. policy shock={bundle.shock_1sd_pp*100:.1f} bps"
        )
        print(
            bundle.peak_summary[
                [
                    "variable",
                    "trough_quarter",
                    "trough_response_per_100bp",
                    "sig_68_negative",
                    "fevd_q4_pct",
                    "fevd_q8_pct",
                    "fevd_q12_pct",
                    "fevd_q20_pct",
                ]
            ].to_string(index=False)
        )

    combined_peaks = pd.concat(all_peaks, ignore_index=True)
    combined_peaks.to_csv(RESULTS_DIR / "all_models_peak_summary.csv", index=False)

    plot_housing_vs_inflation_comparison(bundles[0], bundles[1])
    print(f"\nSaved all SVAR results to {RESULTS_DIR} and figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
