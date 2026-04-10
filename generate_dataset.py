"""
Generate synthetic pharmacy sales data with realistic seasonal patterns.
Enriched dataset includes: store_id, product_name, category, unit_price,
lead_time_days, current_stock, expiry_risk, season, revenue.

Run once: python generate_dataset.py
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    PRODUCTS, SEASON_MAP, SEASONAL_MULTIPLIERS,
    DATA_DIR, DATA_FILE, FESTIVAL_MONTHS,
)

np.random.seed(42)


def generate_sales_data(
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-31",
    store_id: str = "STORE_001",
) -> pd.DataFrame:
    """Generate realistic synthetic pharmacy sales data with enriched columns."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end   = datetime.strptime(end_date,   "%Y-%m-%d")
    num_days = (end - start).days + 1

    # Simulate a running stock ledger per product
    stock_ledger = {pid: info["base_demand"] * 45 for pid, info in PRODUCTS.items()}

    records = []

    for day_offset in range(num_days):
        current_date = start + timedelta(days=day_offset)
        month        = current_date.month
        day_of_week  = current_date.weekday()   # 0=Mon, 6=Sun
        season       = SEASON_MAP[month]
        is_festival  = int(month in FESTIVAL_MONTHS)

        for product_id, info in PRODUCTS.items():
            base     = info["base_demand"]
            category = info["category"]
            lead     = info["lead_time_days"]

            # ── Demand calculation ──────────────────────────────────────
            seasonal_mult = SEASONAL_MULTIPLIERS.get((season, category), 1.0)

            dow_effect = 0.9 if day_of_week >= 5 else 1.0
            if category in ("supplement", "diabetes"):
                dow_effect = 1.0  # chronic — stable

            # Slight long-term growth trend
            trend = 1.0 + (day_offset / num_days) * 0.15

            # Festival month bumps purchasing
            festival_bump = 1.05 if is_festival else 1.0

            # Random noise
            noise = np.random.normal(1.0, 0.12)

            # Disease outbreak spikes
            outbreak_spike = 1.0
            if season == "Monsoon" and category in ("fever", "antibiotic"):
                if np.random.random() < 0.08:
                    outbreak_spike = np.random.uniform(1.5, 2.5)
            if season == "Winter" and category == "respiratory":
                if np.random.random() < 0.06:
                    outbreak_spike = np.random.uniform(1.3, 2.0)

            quantity = int(
                base * seasonal_mult * dow_effect * trend
                * festival_bump * noise * outbreak_spike
            )
            quantity = max(1, quantity)

            # ── Stock simulation ────────────────────────────────────────
            current = stock_ledger[product_id]
            # Restock every lead_time_days with approximate replenishment
            if day_offset % lead == 0:
                restock = int(base * seasonal_mult * lead * 1.2)
                current += restock

            current -= quantity
            current = max(0, current)
            stock_ledger[product_id] = current

            # Expiry risk: flag if stock is more than 60× daily demand
            avg_daily = base * seasonal_mult
            expiry_risk = int(current > avg_daily * 60)

            revenue = round(quantity * info["unit_price"], 2)

            records.append({
                "date":           current_date.strftime("%Y-%m-%d"),
                "store_id":       store_id,
                "product_id":     product_id,
                "product_name":   info["name"],
                "category":       category,
                "season":         season,
                "quantity_sold":  quantity,
                "unit_price":     info["unit_price"],
                "revenue":        revenue,
                "current_stock":  current,
                "lead_time_days": lead,
                "expiry_risk":    expiry_risk,
                "is_festival_month": is_festival,
            })

    df = pd.DataFrame(records)
    return df


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("🏥 Generating enriched synthetic pharmacy sales data...")
    df = generate_sales_data()
    df.to_csv(DATA_FILE, index=False)

    print(f"✅ Dataset saved → {DATA_FILE}")
    print(f"   Rows     : {len(df):,}")
    print(f"   Columns  : {list(df.columns)}")
    print(f"   Date range: {df['date'].min()} → {df['date'].max()}")
    print(f"   Products : {df['product_id'].nunique()}")
    print(f"\n📊 Sample (5 rows):")
    print(df.head(5).to_string(index=False))

    print(f"\n📈 Quantity stats per product:")
    stats = (
        df.groupby("product_name")["quantity_sold"]
        .agg(["mean", "std", "min", "max"])
        .round(1)
    )
    print(stats.to_string())


if __name__ == "__main__":
    main()
