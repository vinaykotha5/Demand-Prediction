"""
Data preprocessing for the LSTM demand forecasting model.
Handles loading, feature engineering, normalization, and sequence creation.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import SEASON_MAP, SEQUENCE_LENGTH, DATA_FILE, SCALER_DIR


def get_season(month: int) -> str:
    """Map month number to Indian season."""
    return SEASON_MAP.get(month, "Unknown")


SEASON_ENCODING = {"Summer": 0, "Monsoon": 1, "Post-Monsoon": 2, "Winter": 3}


def load_and_preprocess(filepath: str = None) -> pd.DataFrame:
    """Load CSV and add time/seasonal features."""
    if filepath is None:
        filepath = DATA_FILE

    df = pd.read_csv(filepath, parse_dates=["date"])
    df.sort_values(["product_id", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Time features
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    # Season features
    df["season"] = df["month"].map(get_season)
    df["season_encoded"] = df["season"].map(SEASON_ENCODING)

    return df


def prepare_product_data(df: pd.DataFrame, product_id: str) -> pd.DataFrame:
    """Filter and prepare data for a single product."""
    product_df = df[df["product_id"] == product_id].copy()
    product_df.sort_values("date", inplace=True)
    product_df.reset_index(drop=True, inplace=True)
    return product_df


def create_features(product_df: pd.DataFrame) -> np.ndarray:
    """Extract feature columns as numpy array."""
    feature_cols = [
        "quantity_sold",
        "month",
        "day_of_week",
        "day_of_month",
        "season_encoded",
    ]
    return product_df[feature_cols].values.astype(np.float32)


def scale_data(
    data: np.ndarray, product_id: str, fit: bool = True
) -> tuple[np.ndarray, MinMaxScaler]:
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
) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding-window sequences for LSTM.

    Returns:
        X: shape (num_samples, seq_length, num_features)
        y: shape (num_samples,) — next-day quantity_sold (column 0)
    """
    if seq_length is None:
        seq_length = SEQUENCE_LENGTH

    X, y = [], []
    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i - seq_length : i])
        y.append(scaled_data[i, 0])  # quantity_sold is column 0

    return np.array(X), np.array(y)


def train_test_split(
    X: np.ndarray, y: np.ndarray, test_ratio: float = 0.2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Chronological train/test split (no shuffling for time-series)."""
    split_idx = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return X_train, X_test, y_train, y_test


def get_full_pipeline(
    product_id: str, filepath: str = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """End-to-end: load → preprocess → features → scale → sequences → split."""
    df = load_and_preprocess(filepath)
    product_df = prepare_product_data(df, product_id)
    features = create_features(product_df)
    scaled, scaler = scale_data(features, product_id, fit=True)
    X, y = create_sequences(scaled)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    return X_train, X_test, y_train, y_test, scaler
