"""
Data preprocessing for the LSTM demand forecasting model.
Handles loading, feature engineering, normalization, and sequence creation.

Feature set (v2):
  Time-based  : month, day_of_week, day_of_month, quarter, is_weekend,
                week_sin, week_cos (cyclical encoding)
  Seasonal    : season_encoded, is_festival_month, is_peak_disease_season
  Lag features: lag_1, lag_7, lag_30
  Rolling stats: rolling_mean_7, rolling_mean_30, rolling_std_7
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import (
    SEASON_MAP, SEQUENCE_LENGTH, DATA_FILE, SCALER_DIR,
    FESTIVAL_MONTHS, PEAK_DISEASE_SEASONS,
)


SEASON_ENCODING = {"Summer": 0, "Monsoon": 1, "Post-Monsoon": 2, "Winter": 3}

# All feature columns fed into the LSTM (order matters — scaler is fit on this)
FEATURE_COLS = [
    "quantity_sold",
    "month",
    "day_of_week",
    "day_of_month",
    "quarter",
    "is_weekend",
    "week_sin",
    "week_cos",
    "season_encoded",
    "is_festival_month",
    "is_peak_disease_season",
    "lag_1",
    "lag_7",
    "lag_30",
    "rolling_mean_7",
    "rolling_mean_30",
    "rolling_std_7",
]


def get_season(month: int) -> str:
    """Map month number to Indian season."""
    return SEASON_MAP.get(month, "Unknown")


def load_and_preprocess(filepath: str = None) -> pd.DataFrame:
    """Load CSV and add all time/seasonal base features (no lags yet)."""
    if filepath is None:
        filepath = DATA_FILE

    df = pd.read_csv(filepath, parse_dates=["date"])
    df.sort_values(["product_id", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── Time features ────────────────────────────────────────────────
    df["month"]        = df["date"].dt.month
    df["day_of_week"]  = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["quarter"]      = df["date"].dt.quarter
    df["is_weekend"]   = (df["date"].dt.dayofweek >= 5).astype(int)

    # Cyclical week-of-year encoding
    week = df["date"].dt.isocalendar().week.astype(float)
    df["week_sin"] = np.sin(2 * np.pi * week / 52.0)
    df["week_cos"] = np.cos(2 * np.pi * week / 52.0)

    # ── Seasonal features ────────────────────────────────────────────
    df["season"]         = df["month"].map(get_season)
    df["season_encoded"] = df["season"].map(SEASON_ENCODING).fillna(0).astype(int)

    # Festival month flag (from config)
    df["is_festival_month"] = df["month"].apply(
        lambda m: 1 if m in FESTIVAL_MONTHS else 0
    )

    return df


def prepare_product_data(df: pd.DataFrame, product_id: str) -> pd.DataFrame:
    """
    Filter data for a single product and compute all lag/rolling features.
    Drops any leading rows where lag/rolling values are NaN.
    """
    product_df = df[df["product_id"] == product_id].copy()
    product_df.sort_values("date", inplace=True)
    product_df.reset_index(drop=True, inplace=True)

    # ── Peak disease season flag ─────────────────────────────────────
    category = product_df["category"].iloc[0] if "category" in product_df.columns else "unknown"
    product_df["is_peak_disease_season"] = product_df.apply(
        lambda r: 1 if (r["season"], category) in PEAK_DISEASE_SEASONS else 0,
        axis=1,
    )

    # ── Lag features ─────────────────────────────────────────────────
    product_df["lag_1"]  = product_df["quantity_sold"].shift(1)
    product_df["lag_7"]  = product_df["quantity_sold"].shift(7)
    product_df["lag_30"] = product_df["quantity_sold"].shift(30)

    # ── Rolling statistics ────────────────────────────────────────────
    product_df["rolling_mean_7"]  = product_df["quantity_sold"].rolling(7,  min_periods=1).mean()
    product_df["rolling_mean_30"] = product_df["quantity_sold"].rolling(30, min_periods=1).mean()
    product_df["rolling_std_7"]   = product_df["quantity_sold"].rolling(7,  min_periods=1).std().fillna(0)

    # Drop rows with NaN lags (first 30 rows per product)
    product_df.dropna(subset=["lag_1", "lag_7", "lag_30"], inplace=True)
    product_df.reset_index(drop=True, inplace=True)

    return product_df


def create_features(product_df: pd.DataFrame) -> np.ndarray:
    """Extract feature columns as numpy array (float32)."""
    # Ensure all feature columns exist
    for col in FEATURE_COLS:
        if col not in product_df.columns:
            product_df[col] = 0
    return product_df[FEATURE_COLS].values.astype(np.float32)


def scale_data(
    data: np.ndarray, product_id: str, fit: bool = True
) -> tuple:
    """Scale features using MinMaxScaler. Save/load scaler per product."""
    os.makedirs(SCALER_DIR, exist_ok=True)
    scaler_path = os.path.join(SCALER_DIR, f"scaler_{product_id}.pkl")

    if fit:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(data)
        joblib.dump(scaler, scaler_path)
    else:
        scaler = joblib.load(scaler_path)
        scaled = scaler.transform(data)

    return scaled, scaler


def create_sequences(
    scaled_data: np.ndarray, seq_length: int = None
) -> tuple:
    """
    Create sliding-window sequences for LSTM.

    Returns:
        X: (num_samples, seq_length, num_features)
        y: (num_samples,) — next-day quantity_sold (column 0)
    """
    if seq_length is None:
        seq_length = SEQUENCE_LENGTH

    X, y = [], []
    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i - seq_length: i])
        y.append(scaled_data[i, 0])   # quantity_sold is column 0

    return np.array(X), np.array(y)


def train_test_split(
    X: np.ndarray, y: np.ndarray, test_ratio: float = 0.2
) -> tuple:
    """Chronological train/test split (no shuffle — time-series safe)."""
    split_idx = int(len(X) * (1 - test_ratio))
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]


def get_full_pipeline(
    product_id: str, filepath: str = None
) -> tuple:
    """
    End-to-end pipeline:
    load → preprocess → lag features → feature matrix → scale → sequences → split

    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    df          = load_and_preprocess(filepath)
    product_df  = prepare_product_data(df, product_id)
    features    = create_features(product_df)
    scaled, scaler = scale_data(features, product_id, fit=True)
    X, y        = create_sequences(scaled)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    return X_train, X_test, y_train, y_test, scaler


def get_xgb_features(product_df: pd.DataFrame) -> tuple:
    """
    Prepare tabular (non-sequence) features for XGBoost baseline.
    Returns (X, y) arrays suitable for sklearn-style models.
    """
    tabular_cols = [
        "month", "day_of_week", "day_of_month", "quarter", "is_weekend",
        "week_sin", "week_cos", "season_encoded",
        "is_festival_month", "is_peak_disease_season",
        "lag_1", "lag_7", "lag_30",
        "rolling_mean_7", "rolling_mean_30", "rolling_std_7",
    ]
    for col in tabular_cols:
        if col not in product_df.columns:
            product_df[col] = 0

    X = product_df[tabular_cols].values.astype(np.float32)
    y = product_df["quantity_sold"].values.astype(np.float32)
    return X, y
