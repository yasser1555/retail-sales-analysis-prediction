# Dataset access

The original and processed CSV files are **intentionally not distributed** here.
The source data is the Kaggle competition [Walmart Recruiting - Store Sales Forecasting](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting/data), marked **Subject to Competition Rules**.

1. Sign in to Kaggle and review/accept the competition rules.
2. Download the `train.csv`, `features.csv`, and `stores.csv` files from its Data tab. Kaggle may provide the first two as nested `.csv.zip` files; extract them.
3. Place the resulting three CSV files at `data/raw/train.csv`, `data/raw/features.csv`, and `data/raw/stores.csv`.
4. Run `python src/prepare_data.py` from the repository root. Output is written to `data/processed/cleaned_retail_data.csv`.

The `data/raw/` and `data/processed/` directories are created locally, and their contents are excluded by `.gitignore`. Do not commit Kaggle data, even if smaller than GitHub's technical file-size limit.
