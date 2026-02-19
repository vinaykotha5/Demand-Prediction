"""
Training orchestration: load data → preprocess → train LSTM per product → save.
Run: python src/train.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from config import PRODUCTS, EPOCHS, BATCH_SIZE, MODEL_DIR
from src.data_preprocessing import get_full_pipeline
from src.model import train_model


def train_all_products(epochs: int = None, batch_size: int = None):
    """Train LSTM models for all products in catalog."""
    if epochs is None:
        epochs = EPOCHS
    if batch_size is None:
        batch_size = BATCH_SIZE

    os.makedirs(MODEL_DIR, exist_ok=True)

    total = len(PRODUCTS)
    print(f"\n{'='*60}")
    print(f"🏋️ Training LSTM models for {total} products")
    print(f"{'='*60}\n")

    results = {}

    for idx, (product_id, info) in enumerate(PRODUCTS.items(), 1):
        print(f"\n[{idx}/{total}] 📦 {info['name']} ({product_id})")
        print("-" * 40)

        start = time.time()

        try:
            X_train, X_test, y_train, y_test, scaler = get_full_pipeline(product_id)

            print(f"   Train: {X_train.shape}, Test: {X_test.shape}")

            model, history = train_model(
                X_train, y_train, product_id,
                epochs=epochs, batch_size=batch_size,
            )

            # Evaluate on test set
            test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
            elapsed = time.time() - start

            results[product_id] = {
                "name": info["name"],
                "test_loss": round(test_loss, 6),
                "test_mae": round(test_mae, 6),
                "train_epochs": len(history.history["loss"]),
                "time_seconds": round(elapsed, 1),
            }

            print(f"   ✅ Test MSE: {test_loss:.6f} | MAE: {test_mae:.6f} | Time: {elapsed:.1f}s")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            results[product_id] = {"name": info["name"], "error": str(e)}

    # Summary
    print(f"\n{'='*60}")
    print("📊 Training Summary")
    print(f"{'='*60}")
    for pid, res in results.items():
        if "error" in res:
            print(f"  ❌ {res['name']}: {res['error']}")
        else:
            print(f"  ✅ {res['name']}: MAE={res['test_mae']:.4f}, "
                  f"Epochs={res['train_epochs']}, Time={res['time_seconds']}s")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    train_all_products(epochs=30, batch_size=32)
