"""Train the final 10-feature Random Forest documented in Walmart_proj.ipynb.

This portable helper follows the *final* notebook model configuration.
Historical metrics belong to the stored notebook outputs, not this file.
Training is not performed merely by downloading the repository.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTENDED_FEATURES = [
    "Store", "Dept", "IsHoliday", "Size", "Type_B", "Type_C",
    "Temperature", "Fuel_Price", "CPI", "Unemployment",
]

def main() -> None:
    parser = argparse.ArgumentParser(description="Train the final retail sales regressor")
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "processed" / "cleaned_retail_data.csv")
    parser.add_argument("--model-dir", type=Path, default=REPO_ROOT / "models")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    if "Type" in df.columns:
        df = pd.get_dummies(df, columns=["Type"], drop_first=True)
    # Already-encoded historical cleaned_retail_data.csv is also accepted.
    missing = [name for name in EXTENDED_FEATURES + ["Weekly_Sales"] if name not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}. Use src/prepare_data.py first.")
    X = df[EXTENDED_FEATURES]
    y = df["Weekly_Sales"]
    if X.isna().any().any() or y.isna().any():
        raise ValueError("Dataset contains missing model features/targets; examine preprocessing.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    mae = mean_absolute_error(y_test, predicted)
    rmse = np.sqrt(mean_squared_error(y_test, predicted))

    args.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_dir / "sales_model.pkl")
    joblib.dump(EXTENDED_FEATURES, args.model_dir / "model_features.pkl")
    print(f"This run's holdout MAE: {mae:.6f}")
    print(f"This run's holdout RMSE: {rmse:.6f}")
    print(f"Saved model and feature list to {args.model_dir}")

if __name__ == "__main__":
    main()
