"""Automated API ingestion for Canadian macroeconomic and monetary policy data.

Fetches keyless public series from:
1. Statistics Canada Web Data Service (WDS) API:
   - Real GDP (Quarterly, Chained 2017 Dollars, SAAR): v62305752 (Table 36-10-0104-01)
   - Headline CPI, All-Items, Canada (Monthly, Seasonally Adjusted): v41690914 (Table 18-10-0006-01)
   - Headline CPI, All-Items, Canada (Monthly, Not Seasonally Adjusted): v41690973 (Table 18-10-0004-01)
   - New Housing Price Index (NHPI), Total, Canada (Monthly): v111955442 (Table 18-10-0205-01)
   - Legacy USD/CAD Monthly Noon Exchange Rate (1970-2017): v37426 (Table 33-10-0163-01)
2. Bank of Canada Valet API:
   - Target for the Overnight Rate (Daily, 2009-present): V39079
   - Overnight Money Market Financing Rate (Monthly, 1975-present): V122514
   - Bank Rate (Monthly, 1935-present): V122530
   - USD/CAD Monthly Average Exchange Rate (2017-present): FXMUSDCAD
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"

STATCAN_WDS_URL = (
    "https://www150.statcan.gc.ca/t1/wds/rest/getDataFromVectorsAndLatestNPeriods"
)
BOC_VALET_URL = "https://www.bankofcanada.ca/valet/observations/{series}/json"

STATCAN_VECTORS: dict[str, dict[str, Any]] = {
    "real_gdp": {
        "vector_id": 62305752,
        "latest_n": 200,
        "description": "Real GDP at market prices, chained 2017 dollars, SAAR (quarterly)",
        "table": "36-10-0104-01",
    },
    "cpi_sa": {
        "vector_id": 41690914,
        "latest_n": 500,
        "description": "CPI All-items, Canada, seasonally adjusted (monthly)",
        "table": "18-10-0006-01",
    },
    "cpi_nsa": {
        "vector_id": 41690973,
        "latest_n": 500,
        "description": "CPI All-items, Canada, not seasonally adjusted (monthly)",
        "table": "18-10-0004-01",
    },
    "nhpi": {
        "vector_id": 111955442,
        "latest_n": 600,
        "description": "New Housing Price Index (NHPI), total house & land, Canada (monthly)",
        "table": "18-10-0205-01",
    },
    "usdcad_legacy": {
        "vector_id": 37426,
        "latest_n": 600,
        "description": "Legacy USD/CAD monthly average noon exchange rate (monthly, pre-2017)",
        "table": "33-10-0163-01",
    },
}

BOC_SERIES: dict[str, dict[str, str]] = {
    "overnight_target": {
        "series_id": "V39079",
        "description": "Bank of Canada target for the overnight rate (daily, 2009-present)",
    },
    "overnight_mmf": {
        "series_id": "V122514",
        "description": "Overnight money market financing rate (monthly, 1990-present)",
    },
    "bank_rate": {
        "series_id": "V122530",
        "description": "Bank of Canada Bank Rate (monthly, reference/fallback)",
    },
    "usdcad_modern": {
        "series_id": "FXMUSDCAD",
        "description": "Bank of Canada USD/CAD monthly average exchange rate (2017-present)",
    },
}


def fetch_statcan_vectors(
    vectors: dict[str, dict[str, Any]], timeout: int = 30
) -> dict[str, pd.DataFrame]:
    """Fetch multiple StatCan vectors in a single batch POST request."""
    payload = [
        {"vectorId": cfg["vector_id"], "latestN": cfg["latest_n"]}
        for cfg in vectors.values()
    ]
    response = requests.post(STATCAN_WDS_URL, json=payload, timeout=timeout)
    response.raise_for_status()
    items = response.json()

    id_to_name = {cfg["vector_id"]: name for name, cfg in vectors.items()}
    results: dict[str, pd.DataFrame] = {}

    for item in items:
        if item.get("status") != "SUCCESS":
            raise RuntimeError(f"StatCan WDS error: {item}")
        obj = item["object"]
        vid = int(obj["vectorId"])
        name = id_to_name[vid]
        points = obj.get("vectorDataPoint", [])
        df = pd.DataFrame(
            [
                {
                    "date": pt["refPer"],
                    "value": pt["value"],
                }
                for pt in points
                if pt.get("value") is not None
            ]
        )
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"]).sort_values("date").reset_index(drop=True)
        results[name] = df

    return results


def fetch_boc_series(
    series_id: str, start_date: str = "1990-01-01", timeout: int = 30
) -> pd.DataFrame:
    """Fetch a single series from the Bank of Canada Valet API."""
    url = BOC_VALET_URL.format(series=series_id)
    response = requests.get(url, params={"start_date": start_date}, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    observations = data.get("observations", [])

    rows = []
    for obs in observations:
        date_str = obs.get("d")
        val_obj = obs.get(series_id)
        if isinstance(val_obj, dict) and val_obj.get("v") not in (None, ""):
            rows.append({"date": date_str, "value": val_obj["v"]})

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No observations returned for BoC series {series_id}")
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).sort_values("date").reset_index(drop=True)
    return df


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("Fetching Statistics Canada vectors...")
    statcan_dfs = fetch_statcan_vectors(STATCAN_VECTORS)

    print("Fetching Bank of Canada Valet series...")
    boc_dfs = {
        name: fetch_boc_series(cfg["series_id"])
        for name, cfg in BOC_SERIES.items()
    }

    all_dfs = {**statcan_dfs, **boc_dfs}
    metadata: dict[str, Any] = {
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "series": {},
    }

    print("\nDownloaded raw series summary:")
    print(f"{'Series':<18} {'Rows':>6} {'Start':<12} {'End':<12}")
    print("-" * 52)
    for name, df in all_dfs.items():
        out_path = RAW_DIR / f"{name}.csv"
        df.to_csv(out_path, index=False)
        start_str = df["date"].min().strftime("%Y-%m-%d")
        end_str = df["date"].max().strftime("%Y-%m-%d")
        print(f"{name:<18} {len(df):>6} {start_str:<12} {end_str:<12}")

        source_cfg = STATCAN_VECTORS.get(name) or BOC_SERIES.get(name) or {}
        metadata["series"][name] = {
            "rows": int(len(df)),
            "start_date": start_str,
            "end_date": end_str,
            "file": out_path.name,
            **source_cfg,
        }

    meta_path = RAW_DIR / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nSaved raw datasets and metadata to {RAW_DIR}")


if __name__ == "__main__":
    main()
