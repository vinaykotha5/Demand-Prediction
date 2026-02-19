"""
Generate synthetic pharmacy sales data with realistic seasonal patterns.
Run this script once to create the dataset: python generate_dataset.py
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PRODUCTS, SEASON_MAP, SEASONAL_MULTIPLIERS, DATA_DIR, DATA_FILE

np.random.seed(42)


def generate_sales_data(
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-31",
    store_id: str = "STORE_001",
) -> pd.DataFrame:
    """Generate realistic synthetic pharmacy sales data."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    num_days = (end - start).days + 1

    records = []

    for day_offset in range(num_days):
        current_date = start + timedelta(days=day_offset)
        month = current_date.month
        day_of_week = current_date.weekday()  # 0=Mon, 6=Sun
        season = SEASON_MAP[month]

        for product_id, info in PRODUCTS.items():
            base = info["base_demand"]
            category = info["category"]

            # Seasonal multiplier
            seasonal_mult = SEASONAL_MULTIPLIERS.get((season, category), 1.0)

            # Day-of-week effect (weekends slightly lower for some categories)
            dow_effect = 0.9 if day_of_week >= 5 else 1.0
            if category in ("supplement", "diabetes"):
                dow_effect = 1.0  # stable demand

            # Monthly micro-trend (slight growth over time)
            trend = 1.0 + (day_offset / num_days) * 0.15

            # Random noise
            noise = np.random.normal(1.0, 0.12)

            # Disease outbreak spikes (random bursts during monsoon for fever/antibiotic)
            outbreak_spike = 1.0
            if season == "Monsoon" and category in ("fever", "antibiotic"):
                if np.random.random() < 0.08:  # 8% chance of spike days
                    outbreak_spike = np.random.uniform(1.5, 2.5)

            # Flu season spikes in winter for respiratory
            if season == "Winter" and category == "respiratory":
                if np.random.random() < 0.06:
                    outbreak_spike = np.random.uniform(1.3, 2.0)

            quantity = int(
                base * seasonal_mult * dow_effect * trend * noise * outbreak_spike
            )
            quantity = max(1, quantity)  # at least 1 unit sold

            records.append(
                {
                    "date": current_date.strftime("%Y-%m-%d"),
                    "product_id": product_id,
                    "product_name": info["name"],
                    "category": category,
                    "quantity_sold": quantity,
                    "unit_price": info["unit_price"],
                    "store_id": store_id,
                }
            )

    df = pd.DataFrame(records)
    return df


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("🏥 Generating synthetic pharmacy sales data...")
    df = generate_sales_data()
    df.to_csv(DATA_FILE, index=False)

    print(f"✅ Dataset saved to {DATA_FILE}")
    print(f"   Rows: {len(df):,}")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Products: {df['product_id'].nunique()}")
    print(f"\n📊 Sample rows:")
    print(df.head(10).to_string(index=False))

    print(f"\n📈 Quantity stats per product:")
    stats = (
        df.groupby("product_name")["quantity_sold"]
        .agg(["mean", "std", "min", "max"])
        .round(1)
    )
    print(stats.to_string())


if __name__ == "__main__":
    main()
