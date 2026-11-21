# Retail Sales Analysis & Prediction

**Exploratory data analysis, regression benchmarking, and a Streamlit interface for department-level weekly retail sales predictions.**

> Academic **team** AI / Data Science project presented in a personal portfolio. The notebook outputs document completed experiments; the repository does **not** claim independent implementation by the portfolio owner, production readiness, a Kaggle competition score, or an independently reproduced evaluation.

## 1. Project Overview

This project explores historical weekly sales for Walmart stores and departments, combines sales records with store attributes and external economic indicators, and trains regression models to estimate `Weekly_Sales`. The supplied academic notebook contains exploratory charts, a constant-mean baseline, Linear Regression, and two Random Forest configurations. A Streamlit form uses the **final 10-feature Random Forest** to make one prediction at a time when its serialized artifacts are available locally.

## 2. Problem Statement & Objectives

Historical demand varies across stores, departments and holiday periods. The completed work aims to:

- identify store-level, departmental, seasonal and holiday-associated sales patterns;
- clean and join sales, store, and contextual economic data;
- compare a simple baseline and regression models using the recorded metrics;
- expose the selected model through a lightweight interactive application.

The model performs **point predictions from supplied input features**. Although the dataset includes weekly dates for analysis, the trained regressors do not use date-derived predictors or a chronological validation split; this is **not a validated forward-looking time-series forecast**.

## 3. Dataset Overview

**Source:** [Walmart Recruiting - Store Sales Forecasting, Kaggle](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting/data). The competition data is **subject to its competition rules**; no competition data is redistributed in this repository.

| Original file | Contents | Rows observed in the supplied project |
| --- | --- | ---: |
| `train.csv` | Store, department, date, weekly sales, holiday flag | 421,570 |
| `features.csv` | Temperature, fuel price, promotional markdown fields, CPI, unemployment, holiday flag | 8,190 |
| `stores.csv` | Store identifier, store type, store size | 45 |

After excluding **1,285 records with negative weekly sales**, the merged analysis dataset contained **420,285 rows**, covering **2010-02-05 to 2012-10-26**. The target is `Weekly_Sales` for a specific store/department/week; sales are presented in the source's dollar-denominated format.

**To obtain the data:** sign in to Kaggle, review and accept the competition rules, download/extract `train.csv`, `features.csv`, and `stores.csv` from the [competition Data page](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting/data), and put them in `data/raw/`. See [`data/README.md`](data/README.md). The notebook originally used Colab's file-upload dialog; the new CLI helper uses the structured paths shown here.

## 4. Technologies & Tools

- **Python** and **Jupyter Notebook / Google Colab** for the academic workflow.
- **pandas, NumPy** for tabular joins, cleaning, one-hot encoding and calculations.
- **Matplotlib** for exploratory charts and feature importance plots.
- **scikit-learn** for train/test split, Linear Regression, Random Forest Regressor, MAE and RMSE.
- **joblib** for local model and feature-list serialization.
- **Streamlit** for the interactive prediction interface.

Exact versions of the libraries used in the original academic run are not recorded; `requirements.txt` lists the actual external Python libraries, without claiming historical version pins.

## 5. Data Preprocessing & Exploratory Analysis

The original notebook checked missingness and duplicates in the three raw tables (zero duplicates reported), removed negative sales rows, parsed dates, left-joined sales with the contextual features on `Store` + `Date` and with the store table on `Store`, resolved duplicate `IsHoliday` columns, and removed `MarkDown1`–`MarkDown5` because they contained many missing values. Its merged training-period data had no remaining nulls. It encoded `Type` into `Type_B` and `Type_C` using `pd.get_dummies(..., drop_first=True)` for model training.

**EDA results:** recorded plots show substantial differences in aggregate sales by store, higher average weekly sales for Type A stores, a broad positive association between store size and sales, and holiday/seasonal spikes. The following are **plots saved within the original notebook**, exported without redrawing or changing data:

| Plot | Preview |
| --- | --- |
| Weekly sales trend | ![Recorded weekly trend](figures/05_total_weekly_sales_over_time.png) |
| Average sales by store type | ![Recorded store type analysis](figures/03_average_sales_by_store_type.png) |
| Holiday vs non-holiday | ![Recorded holiday comparison](figures/06_holiday_vs_nonholiday_sales.png) |
| Model feature importance | ![Recorded feature importance](figures/08_extended_model_feature_importance.png) |

Additional original charts are in [`figures/`](figures/). Note that feature importance reflects the fitted model, not a causal interpretation of economic conditions.

## 6. Machine Learning Models

| Method | Input columns / configuration |
| --- | --- |
| Mean baseline | Constant `Weekly_Sales` mean (computed on the entire cleaned dataset in the notebook) |
| Linear Regression | `Store`, `Dept`, `IsHoliday`, `Size`, `Type_B`, `Type_C` |
| Random Forest, original | Same six features; `n_estimators=50`, `random_state=42` |
| Random Forest, extended | Above six **plus** `Temperature`, `Fuel_Price`, `CPI`, `Unemployment`; `n_estimators=50`, `random_state=42` |

The three trained regressors used `train_test_split(test_size=0.2, random_state=42)`. The extended Random Forest was saved to `sales_model.pkl` and its ordered 10-feature list to `model_features.pkl` in the archived notebook.

## 7. Model Evaluation & Actual Results

**Values from saved notebook outputs (not rerun):**

| Model | MAE | RMSE |
| --- | ---: | ---: |
| Mean baseline | 15,177.28 | Not recorded |
| Linear Regression | 14,614.45 | Not recorded |
| Random Forest (6 features) | 2,713.10 | 7,349.40 |
| **Random Forest (10 features)** | **1,914.52** | **5,483.60** |

**R² was not reported**. The baseline was evaluated on the full cleaned dataset, unlike the train/test models. The random split can share adjacent dates and familiar store-department pairs between train and test, so these errors should not be interpreted as verified future-period performance. The Kaggle competition uses holiday-weighted MAE (WMAE); **no WMAE or leaderboard submission score is claimed**. The raw stored values and method caveats are in [`results/recorded_metrics.md`](results/recorded_metrics.md).

## 8. Streamlit Application

The application in [`app/app.py`](app/app.py) accepts store ID, department ID, holiday flag, store type, store size, temperature, fuel price, CPI, and unemployment. It orders the 10 derived numeric/binary inputs to match `model_features.pkl`, loads the local trained forest, and displays a weekly-sales estimate plus a heuristic message for low, medium or high demand (thresholds **$10,000** and **$30,000**). These messages are simple app rules, not a separately evaluated inventory optimization system.

**Important:** the original serialized model is approximately **1.52 GB uncompressed** and is **not included**. Until you train a local replacement or supply your own original trusted artifacts in `models/`, the app clearly reports the missing files rather than producing predictions. The app's model-dependent execution was not validated as part of this repository packaging.

## 9. Project Structure

```text
README.md
requirements.txt
.gitignore
notebooks/
  Walmart_proj.ipynb           # Original academic notebook and saved outputs
src/
  prepare_data.py              # Repo helper: original merge/cleaning steps
  train_model.py               # Repo helper: final 10-feature Random Forest
app/
  app.py                       # Original Streamlit logic; portable file loading
figures/                       # 8 figures exported from recorded notebook outputs
results/
  recorded_metrics.md          # Source-grounded metrics and qualifications
data/
  README.md                    # Competition data access instructions
models/
  README.md                    # Local-only artifact instructions
```

`data/raw/`, `data/processed/`, the serialized model and local environments are excluded from version control; no empty placeholder directories are included.

## 10. Installation & Usage

Recommended: Python 3.10+ with enough local memory and storage to train a 50-tree Random Forest on about 420,000 rows. The original package versions are unknown, so exact metric reproduction across library versions is **not guaranteed**.

```bash
git clone https://github.com/yasser1555/retail-sales-analysis-prediction.git
cd retail-sales-analysis-prediction
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

1. Download and extract the three competition CSVs from Kaggle into `data/raw/` (see [`data/README.md`](data/README.md)).
2. Prepare the data:

   ```bash
   python src/prepare_data.py
   ```

   Output: `data/processed/cleaned_retail_data.csv` (with the original `Type` column, prior to one-hot encoding).

3. Train the final model **locally** (resource-intensive; creates large files):

   ```bash
   python src/train_model.py
   ```

   This script follows the final recorded model configuration. Its fresh metric printout, if run by you, is **not** automatically identical to the archived results. It writes `models/sales_model.pkl` and `models/model_features.pkl`. If you have the trusted archived artifacts already, you can place both there instead of retraining.

4. Start Streamlit:

   ```bash
   streamlit run app/app.py
   ```

The untouched original notebook is in [`notebooks/Walmart_proj.ipynb`](notebooks/Walmart_proj.ipynb). It was written for Colab, uses interactive upload/download cells, and refers to `train.csv`, `features.csv`, `stores.csv` and `cleaned_retail_data.csv` in its **working directory**. For re-executing the original notebook rather than the portable CLI helpers, use the original Colab upload steps or place your locally downloaded CSVs where the notebook expects them. Do not commit those files. The provided archived cleaned CSV had already been one-hot encoded, so rerunning the original notebook from that particular CSV would fail when it tries to encode the absent `Type` column again.

## 11. Limitations & Future Improvements

- **Validation:** a chronological train/validation/test split, and preferably backtesting by week/store, is needed to assess actual future-week forecasting.
- **Metrics:** record WMAE to match the Kaggle objective, plus R² and segment/holiday error analyses if run in future experiments.
- **Features:** the models did not use engineered lag, rolling-window, month, or explicit holiday-season date predictors; the original EDA used dates but training did not.
- **Baseline comparability:** re-evaluate the baseline on exactly the same held-out rows as the other models.
- **Data decisions:** the original notebook excluded negative sales and promotional markdown features; explore whether these represent returns or important event information before making different production assumptions.
- **Practical deployment:** the large pickled forest, missing version lockfile and pickle version sensitivity complicate portable/cloud deployment. Do not load untrusted pickle files.
- **Business decisions:** the app's inventory/staffing advice is threshold-based, not experimentally validated.

These are proposed improvements only; **no implementation or claimed metric is implied**.

## 12. Academic Team & Portfolio Attribution

**Yasser Alassad** — Computer Science student specializing in Artificial Intelligence, Effat University; **member of the academic project group and maintainer of this portfolio showcase**.

- [GitHub](https://github.com/yasser1555)
- [LinkedIn](https://www.linkedin.com/in/yasser-alassad/)
- [Portfolio](https://yasser-alassad-portfolio.vercel.app/)

**Team-project disclosure:** The original academic project was presented by a group of three students. Yasser participated as a group member, but **does not claim to have independently developed the code, notebook, Streamlit app, figures, or model experiments**. Individual technical contributions have not been documented in this showcase, so no specific code authorship is attributed here. The work should be credited to the academic team rather than presented as a solo implementation. Obtain the necessary permission from other contributors before publishing shared project code or materials.

**Data attribution:** Walmart Recruiting - Store Sales Forecasting, Kaggle competition (2014); access and permitted uses are governed by the [competition rules](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting/rules). No software license has been asserted for the shared academic source material.
