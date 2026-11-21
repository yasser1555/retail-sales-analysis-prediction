"""Reproduce the cleaning/merging sequence in the archived Colab notebook.

This helper was organized for the public repository; the notebook is the
original record of the analysis. It does not perform model training.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_COLUMNS = [f"MarkDown{i}" for i in range(1, 6)]

def prepare(raw_dir: Path, output: Path) -> pd.DataFrame:
    train = pd.read_csv(raw_dir / "train.csv")
    features = pd.read_csv(raw_dir / "features.csv")
    stores = pd.read_csv(raw_dir / "stores.csv")

    # As in the notebook: exclude negative observations, do not
    # substitute them with zero or modify positive sales amounts.
    train = train[train["Weekly_Sales"] >= 0].copy()
    train["Date"] = pd.to_datetime(train["Date"])
    features["Date"] = pd.to_datetime(features["Date"])

    df = pd.merge(train, features, on=["Store", "Date"], how="left")
    df = pd.merge(df, stores, on="Store", how="left")
    df = df.drop(columns=["IsHoliday_y"])
    df = df.rename(columns={"IsHoliday_x": "IsHoliday"})
    df = df.drop(columns=MARKDOWN_COLUMNS)

    # Keep Type as A/B/C here: the notebook encodes it later, just
    # before fitting. This avoids re-encoding an already encoded CSV.
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Saved {len(df):,} rows and {len(df.columns)} columns to {output}")
    if df.isna().any().any():
        print("Warning: missing values remain in the processed data.")
    return df

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Walmart retail sales data")
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data" / "processed" / "cleaned_retail_data.csv")
    args = parser.parse_args()
    prepare(args.raw_dir, args.output)

if __name__ == "__main__":
    main()
