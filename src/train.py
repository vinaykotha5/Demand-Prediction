"""
Training orchestration for PharmaCast — v2.
Trains LSTM per product, evaluates against Naive + XGBoost baselines,
and saves all results to data/training_results.json.

Usage:
    python src/train.py
    python src/train.py --epochs 30
    python src/train.py --epochs 20 --products P001,P002,P005
"""
import os
import sys
import time
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from config import PRODUCTS, EPOCHS, BATCH_SIZE, MODEL_DIR, TRAINING_RESULTS_FILE, DATA_DIR
from src.data_preprocessing import get_full_pipeline, get_xgb_features, load_and_preprocess, prepare_product_data
from src.model import train_model
from src.evaluator import compute_metrics, baseline_naive, baseline_xgboost, compare_models


def train_all_products(
    epochs: int = None,
    batch_size: int = None,
    product_ids: list = None,
):
    """
    Train LSTM models for all (or selected) products.
    After training, runs full model comparison and saves results.
    """
    if epochs     is None: epochs     = EPOCHS
    if batch_size is None: batch_size = BATCH_SIZE
    if product_ids is None: product_ids = list(PRODUCTS.keys())

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DATA_DIR,  exist_ok=True)

    total = len(product_ids)
    print(f"\n{'='*60}")
    print(f"🏋️  Training LSTM models for {total} products")
    print(f"    Epochs: {epochs} | Batch size: {batch_size}")
    print(f"{'='*60}\n")

    try:
        from tqdm import tqdm
        iterator = tqdm(enumerate(product_ids, 1), total=total, desc="Products")
    except ImportError:
        iterator = enumerate(product_ids, 1)

    train_results = {}

    for idx, product_id in iterator:
        info = PRODUCTS.get(product_id, {})
        print(f"\n[{idx}/{total}] 📦 {info.get('name', product_id)} ({product_id})")
        print("-" * 50)

        start = time.time()

        try:
            X_train, X_test, y_train, y_test, scaler = get_full_pipeline(product_id)
            print(f"   Sequences — Train: {X_train.shape}, Test: {X_test.shape}")

            model, history = train_model(
                X_train, y_train, product_id,
                epochs=epochs, batch_size=batch_size,
            )

            # LSTM test evaluation
            test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
            elapsed = time.time() - start

            train_results[product_id] = {
                "name":          info.get("name", product_id),
                "test_mse":      round(test_loss, 6),
                "test_mae_scaled": round(test_mae, 6),
                "train_epochs":  len(history.history["loss"]),
                "time_seconds":  round(elapsed, 1),
                "status":        "trained",
            }

            print(f"   ✅ MSE: {test_loss:.6f} | MAE (scaled): {test_mae:.6f} | Time: {elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - start
            print(f"   ❌ Training error: {e}")
            train_results[product_id] = {
                "name":   info.get("name", product_id),
                "error":  str(e),
                "status": "failed",
                "time_seconds": round(elapsed, 1),
            }

    # ── Compare all models on real-scale metrics ──────────────────────
    print(f"\n{'='*60}")
    print("🔬 Running model comparison (Naive vs XGBoost vs LSTM)...")
    print(f"{'='*60}")

    comparison_results = {}
    for product_id in product_ids:
        try:
            print(f"   📊 {PRODUCTS.get(product_id, {}).get('name', product_id)}")
            comp = compare_models(product_id)
            comparison_results[product_id] = comp

            # Merge into train_results
            if product_id in train_results:
                train_results[product_id]["comparison"] = {
                    "Naive":   comp.get("Naive"),
                    "XGBoost": comp.get("XGBoost"),
                    "LSTM":    comp.get("LSTM"),
                }
        except Exception as e:
            print(f"   ⚠️  Comparison failed for {product_id}: {e}")

    # ── Save results ──────────────────────────────────────────────────
    output_path = TRAINING_RESULTS_FILE
    with open(output_path, "w") as f:
        json.dump(train_results, f, indent=2)
    print(f"\n✅ Training results saved → {output_path}")

    # ── Summary table ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("📊 Training Summary")
    print(f"{'='*60}")
    for pid, res in train_results.items():
        if res.get("status") == "failed":
            print(f"  ❌ {res['name']}: {res.get('error', 'unknown error')}")
        else:
            comp_info = res.get("comparison", {})
            lstm_m    = comp_info.get("LSTM") or {}
            xgb_m     = comp_info.get("XGBoost") or {}
            naive_m   = comp_info.get("Naive") or {}
            print(
                f"  ✅ {res['name']:25s} | "
                f"Naive MAE={naive_m.get('MAE', 'N/A'):<8} | "
                f"XGB MAE={xgb_m.get('MAE', 'N/A'):<8} | "
                f"LSTM MAE={lstm_m.get('MAE', 'N/A'):<8} | "
                f"Epochs={res.get('train_epochs', '?')} | "
                f"Time={res.get('time_seconds', '?')}s"
            )
    print(f"{'='*60}\n")

    return train_results


# ─── CLI entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PharmaCast LSTM models")
    parser.add_argument("--epochs",   type=int,   default=None,  help="Number of training epochs")
    parser.add_argument("--products", type=str,   default=None,  help="Comma-separated product IDs, e.g. P001,P002")
    parser.add_argument("--batch",    type=int,   default=None,  help="Batch size")
    args = parser.parse_args()

    product_list = None
    if args.products:
        product_list = [p.strip() for p in args.products.split(",")]
        invalid = [p for p in product_list if p not in PRODUCTS]
        if invalid:
            print(f"⚠️  Unknown product IDs: {invalid}")
            product_list = [p for p in product_list if p in PRODUCTS]

    train_all_products(
        epochs=args.epochs or 30,
        batch_size=args.batch,
        product_ids=product_list,
    )
