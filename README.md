# Retail Demand Forecasting — End-to-End ML System

An end-to-end proof-of-concept machine learning system that forecasts daily retail demand at the Store × Product level and surfaces predictions to inventory managers through an interactive Streamlit dashboard backed by AWS.

**Business problem:** Retail inventory managers need to know which SKUs are at risk of stocking out before the next delivery. This system uses XGBoost to predict daily units sold, stores results in RDS MySQL, and displays a live Forecast Explorer and Inventory Alert dashboard.

---

## Repository Structure

```
retail-store-inventory-forecasting/
├── [EDA+model selection] retail_inventory.ipynb   # EDA, Prophet vs XGBoost comparison, hyperparameter search
├── [model on sagemaker] sagemaker_retail_xgb.ipynb # XGBoost with best params → writes to RDS
├── streamlit_app.py                                # Streamlit dashboard (Forecast Explorer + Inventory Alerts)
├── requirements.txt                               # Python dependencies for Streamlit Cloud
├── .streamlit/
│   └── secrets.toml                               # RDS credentials — NEVER commit this file
└── README.md
```

---

## Pipeline Overview

```
S3 (CSV)
  ↓
SageMaker Notebook
  · Feature engineering (lag, rolling, calendar features)
  · XGBoost inference with best hyperparameters
  · Writes actuals + predictions to RDS
  ↓
RDS MySQL (2 tables: actuals, predictions)
  ↓
Streamlit App (Forecast Explorer + Inventory Alerts)
```

---

## Notebooks

### `[EDA+model selection] retail_inventory.ipynb`
Run this first to understand the data and reproduce the model selection process. **Does not require AWS.**

- Exploratory data analysis: distributions, correlation heatmap, seasonality, promotion effects
- Data leakage analysis: identifies `Demand Forecast` (r = 1.00 with target) as a leakage column
- Feature engineering: lag features, rolling statistics, calendar features
- Hyperparameter tuning: RandomizedSearchCV + TimeSeriesSplit (20 trials) — best params used in SageMaker notebook
- Model comparison: XGBoost vs. Prophet on a chronological held-out test set
- Prophet component decomposition plots (trend, weekly seasonality, regressor effects)

**Final results (held-out test set, Nov–Dec 2023):**

| Model | RMSE | MAE | Train time |
|---|---|---|---|
| **XGBoost** | **107.89** | **88.68** | 18.0s |
| Prophet | 109.05 | 89.45 | 32.6s |

XGBoost wins by 1.1% RMSE and trains 1.8× faster. Best hyperparameters from this search are hardcoded into the SageMaker notebook.

### `[model on sagemaker] sagemaker_retail_xgb.ipynb`
Run this on AWS SageMaker to generate predictions and populate RDS. Uses the best hyperparameters found in the EDA notebook — **no tuning step, just fit and predict.**

- Reads `retail_store_inventory.csv` directly from S3 via `boto3`
- Applies the same feature engineering as the EDA notebook
- Trains XGBoost with best hyperparameters: `n_estimators=300, learning_rate=0.01, max_depth=6, subsample=0.9, colsample_bytree=0.7, min_child_weight=1`
- Writes two tables to RDS: `actuals` and `predictions` (with 80% CI bounds)
- Includes sanity-check read-back and the SQL queries used by the Streamlit app

---

## Setup

### Prerequisites

- Python 3.10+
- An AWS account with:
  - An S3 bucket containing `retail_store_inventory.csv`
  - An RDS MySQL instance (free tier `db.t3.micro` works fine)
  - A SageMaker notebook instance with an IAM role that has `AmazonS3ReadOnlyAccess` and network access to RDS

### 1. Install dependencies (local / Streamlit Cloud)

```bash
pip install -r requirements.txt
```

### 2. Configure RDS credentials

Create `.streamlit/secrets.toml` (never commit this file):

```toml
[rds]
host     = "YOUR-RDS-ENDPOINT.rds.amazonaws.com"
port     = 3306
db       = "retail_forecast"
user     = "admin"
password = "YOUR-PASSWORD"
```

For Streamlit Cloud, paste the same content into **Settings → Secrets** in your app dashboard instead of using the file.

### 3. Upload data to S3

Upload `retail_store_inventory.csv` to your S3 bucket. Then update the config cell at the top of the SageMaker notebook:

```python
S3_BUCKET   = 'your-bucket-name'
S3_DATA_KEY = 'retail_store_inventory.csv'

RDS_HOST     = 'YOUR-RDS-ENDPOINT.rds.amazonaws.com'
RDS_PORT     = '3306'
RDS_DB       = 'retail_forecast'
RDS_USER     = 'admin'
RDS_PASSWORD = 'YOUR-PASSWORD'
```

### 4. Run the SageMaker notebook

Open `[model on sagemaker] sagemaker_retail_xgb.ipynb` in SageMaker Studio or a Classic Notebook instance and run all cells. The notebook will:

1. Load data from S3
2. Engineer features and split train/test at `2023-11-01`
3. Train XGBoost with pre-tuned best hyperparameters
4. Write predictions and actuals to RDS (creates tables automatically on first run)
5. Print a per-store sanity-check summary

Expected runtime: ~25–40 seconds on `ml.m5.large`.

### 5. Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

Or deploy to Streamlit Community Cloud by connecting your GitHub repo and setting secrets as described in step 2.

---

## RDS Schema

Two tables are created automatically by the SageMaker notebook:

| Table | Primary Key | Description |
|---|---|---|
| `actuals` | `(store_id, product_id, obs_date)` | Ground truth daily sales with price, discount, promotion flag |
| `predictions` | `(store_id, product_id, obs_date, run_timestamp)` | XGBoost forecasts with 80% CI bounds and absolute error |

The Streamlit app always queries `WHERE run_timestamp = (SELECT MAX(run_timestamp) FROM predictions)` so re-running the SageMaker notebook refreshes predictions without breaking the dashboard.

---

## Key Design Decisions

**No data leakage.** `Demand Forecast`, `Inventory Level`, and `Units Ordered` are excluded from model features. `Demand Forecast` has r = 1.00 with `Units Sold` — including it gives the model the answer directly and produces inflated R² ≥ 0.99.

**Chronological splitting.** Train/test split at `2023-11-01`. No random shuffling at any stage. Hyperparameter tuning uses `TimeSeriesSplit(n_splits=3)` to respect temporal ordering.

**Hyperparameters pre-tuned, not re-tuned on SageMaker.** The EDA notebook runs `RandomizedSearchCV` (20 trials) to find the best parameters. Those are hardcoded in the SageMaker notebook to keep inference fast and reproducible.

**Residual-based confidence intervals.** XGBoost does not produce native prediction intervals. The 80% CI is estimated from the 10th and 90th percentiles of per-record residuals on the test set.

**Days of supply.** Inventory alerts use `inventory_level ÷ avg_predicted_demand` — current stock divided by the model's average daily forecast — giving managers an actionable "days until stockout" number.

---

## Security Notes

- Never commit `.streamlit/secrets.toml`. It is listed in `.gitignore`.
- The RDS credentials in the SageMaker notebook config cell are for POC convenience only. In production, use AWS Secrets Manager or SageMaker environment variables.
- Restrict the RDS security group inbound rule (port 3306) to your SageMaker VPC and Streamlit Cloud's IP range.

---

## Dataset

[Retail Store Inventory Forecasting Dataset](https://www.kaggle.com/datasets/retail-store-inventory-forecasting-dataset) — Kaggle, 2024.  
73,100 rows · 15 columns · 5 stores · 20 products · 2 years (2022–2023)
