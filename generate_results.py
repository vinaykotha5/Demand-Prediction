"""
One-shot script: generates training_results.json from already-trained models.
Run AFTER src/train.py has saved all .keras files.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from config import PRODUCTS, MODEL_DIR, DATA_DIR, TRAINING_RESULTS_FILE
from src.evaluator import compare_models

os.makedirs(DATA_DIR, exist_ok=True)

results = {}
product_ids = list(PRODUCTS.keys())

print(f"\n{'='*60}")
print(f"🔬 Running model comparison for {len(product_ids)} products...")
print(f"{'='*60}\n")

for pid in product_ids:
    info = PRODUCTS[pid]
    name = info.get("name", pid)
    model_path = os.path.join(MODEL_DIR, f"lstm_{pid}.keras")

    if not os.path.exists(model_path):
        print(f"  ⚠️  No saved model for {name} ({pid}), skipping.")
        results[pid] = {"name": name, "status": "not_trained"}
        continue

    print(f"  📊 {name} ({pid})")
    try:
        t0 = time.time()
        comp = compare_models(pid)
        elapsed = round(time.time() - t0, 1)

        lstm_m  = comp.get("LSTM")    or {}
        xgb_m   = comp.get("XGBoost") or {}
        naive_m = comp.get("Naive")   or {}

        results[pid] = {
            "name":       name,
            "status":     "trained",
            "time_seconds": elapsed,
            "comparison": {
                "Naive":   naive_m,
                "XGBoost": xgb_m,
                "LSTM":    lstm_m,
            },
        }
        print(f"     Naive MAE={naive_m.get('MAE','N/A')}  XGB MAE={xgb_m.get('MAE','N/A')}  LSTM MAE={lstm_m.get('MAE','N/A')}")

    except Exception as e:
        print(f"  ❌ Error for {name}: {e}")
        results[pid] = {"name": name, "status": "comparison_failed", "error": str(e)}

with open(TRAINING_RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ training_results.json saved → {TRAINING_RESULTS_FILE}")
print(f"{'='*60}\n")
