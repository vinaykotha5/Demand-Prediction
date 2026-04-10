"""
Model evaluation module for PharmaCast.

Computes MAE, RMSE, MAPE, Directional Accuracy and
compares three forecast methods:
  1. Naive (previous-day / last-known value)
  2. XGBoost (tabular regressor on engineered features)
  3. LSTM   (from saved model)
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import (
    PRODUCTS, MODEL_DIR, SCALER_DIR, DATA_FILE,
    SEQUENCE_LENGTH, TRAINING_RESULTS_FILE,
)


# ─── Core metrics ─────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute MAE, RMSE, MAPE, and Directional Accuracy.

    Args:
        y_true: actual values
        y_pred: predicted values

    Returns:
        dict with MAE, RMSE, MAPE, DirectionalAccuracy
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # Avoid division by zero in MAPE
    nonzero = y_true != 0
    if nonzero.sum() > 0:
        mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)
    else:
        mape = float("nan")

    # Directional accuracy: % of times prediction moves in same direction as actual
    if len(y_true) > 1:
        actual_dir = np.sign(np.diff(y_true))
        pred_dir   = np.sign(np.diff(y_pred))
        dir_acc    = float(np.mean(actual_dir == pred_dir) * 100)
    else:
        dir_acc = float("nan")

    return {
        "MAE":                 round(mae,     4),
        "RMSE":                round(rmse,    4),
        "MAPE":                round(mape,    2),
        "DirectionalAccuracy": round(dir_acc, 2),
    }


# ─── Baseline models ───────────────────────────────────────────────────

def baseline_naive(y_train: np.ndarray, n_test: int) -> np.ndarray:
    """
    Naive forecast: predict the last known training value for every test step.

    Args:
        y_train: training actuals
        n_test : number of test periods to predict

    Returns:
        predictions array of length n_test
    """
    last_value = y_train[-1]
    return np.full(n_test, last_value)


def baseline_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
) -> np.ndarray:
    """
    XGBoost regressor baseline on tabular engineered features.

    Args:
        X_train: (n_train, n_features) tabular features
        y_train: (n_train,) target values
        X_test : (n_test,  n_features) tabular features

    Returns:
        predictions array of length n_test
    """
    try:
        from xgboost import XGBRegressor
    except ImportError:
        raise ImportError("xgboost not installed. Run: pip install xgboost")

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return np.maximum(preds, 0).astype(float)


# ─── Full comparison pipeline ──────────────────────────────────────────

def compare_models(product_id: str, filepath: str = None) -> dict:
    """
    Train and evaluate Naive, XGBoost, and LSTM (if available) for a product.

    Args:
        product_id: e.g. "P001"
        filepath  : optional path to CSV (defaults to DATA_FILE)

    Returns:
        dict: {
          "product_id": ...,
          "product_name": ...,
          "Naive":    {MAE, RMSE, MAPE, DirectionalAccuracy},
          "XGBoost":  {MAE, RMSE, MAPE, DirectionalAccuracy},
          "LSTM":     {MAE, RMSE, MAPE, DirectionalAccuracy} or None,
        }
    """
    from src.data_preprocessing import (
        load_and_preprocess, prepare_product_data,
        create_features, scale_data, create_sequences,
        train_test_split as tts, get_xgb_features,
    )

    df          = load_and_preprocess(filepath)
    product_df  = prepare_product_data(df, product_id)
    product_name = PRODUCTS.get(product_id, {}).get("name", product_id)

    # ── Tabular split for Naive + XGBoost ────────────────────────────
    X_tab, y_tab = get_xgb_features(product_df)
    split_idx    = int(len(X_tab) * 0.8)
    X_tr, X_te   = X_tab[:split_idx], X_tab[split_idx:]
    y_tr, y_te   = y_tab[:split_idx], y_tab[split_idx:]

    # Naive
    naive_preds  = baseline_naive(y_tr, len(y_te))
    naive_metrics = compute_metrics(y_te, naive_preds)

    # XGBoost
    try:
        xgb_preds    = baseline_xgboost(X_tr, y_tr, X_te)
        xgb_metrics  = compute_metrics(y_te, xgb_preds)
    except Exception as e:
        xgb_metrics  = {"error": str(e)}

    # ── LSTM (if model exists) ────────────────────────────────────────
    lstm_metrics = None
    try:
        import joblib
        from src.model import load_trained_model, predict_demand

        features      = create_features(product_df)
        scaled, scaler = scale_data(features, product_id, fit=False)
        X_seq, y_seq  = create_sequences(scaled)
        _, X_te_seq, _, y_te_seq = tts(X_seq, y_seq)

        model = load_trained_model(product_id)
        y_pred_scaled = model.predict(X_te_seq, verbose=0).flatten()

        # Inverse-transform predictions
        n_features = scaler.n_features_in_
        dummy = np.zeros((len(y_pred_scaled), n_features))
        dummy[:, 0] = y_pred_scaled
        lstm_pred_raw = scaler.inverse_transform(dummy)[:, 0]

        dummy_true = np.zeros((len(y_te_seq), n_features))
        dummy_true[:, 0] = y_te_seq
        y_te_raw = scaler.inverse_transform(dummy_true)[:, 0]

        lstm_metrics = compute_metrics(y_te_raw, lstm_pred_raw)
    except FileNotFoundError:
        lstm_metrics = {"note": "Model not trained yet. Run: python src/train.py"}
    except Exception as e:
        lstm_metrics = {"error": str(e)}

    return {
        "product_id":   product_id,
        "product_name": product_name,
        "test_samples": len(y_te),
        "Naive":    naive_metrics,
        "XGBoost":  xgb_metrics,
        "LSTM":     lstm_metrics,
    }


def evaluate_all_products(filepath: str = None) -> dict:
    """
    Run model comparison for every product and save to TRAINING_RESULTS_FILE.

    Returns:
        dict: {product_id: compare_models() output}
    """
    results = {}
    for product_id in PRODUCTS:
        print(f"  📊 Evaluating {product_id} — {PRODUCTS[product_id]['name']}")
        try:
            results[product_id] = compare_models(product_id, filepath)
        except Exception as e:
            results[product_id] = {"product_id": product_id, "error": str(e)}

    os.makedirs(os.path.dirname(TRAINING_RESULTS_FILE), exist_ok=True)
    with open(TRAINING_RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Evaluation results saved → {TRAINING_RESULTS_FILE}")
    return results


def load_training_results() -> dict:
    """Load saved training/evaluation results from JSON, or return empty dict."""
    if not os.path.exists(TRAINING_RESULTS_FILE):
        return {}
    with open(TRAINING_RESULTS_FILE, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    print("🔬 Running model comparison for all products...")
    evaluate_all_products()
