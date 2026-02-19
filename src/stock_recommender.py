"""
Stock recommendation engine and alert system.
"""
import sys
import os
from datetime import datetime
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import (
    SEASON_MAP,
    SEASONAL_MULTIPLIERS,
    STOCK_BUFFER,
    LOW_STOCK_THRESHOLD,
    OVERSTOCK_THRESHOLD,
    PRODUCTS,
)


def get_season(date_input) -> str:
    """Determine Indian season from a date.

    Args:
        date_input: str (YYYY-MM-DD), datetime, or int (month number)
    """
    if isinstance(date_input, str):
        month = datetime.strptime(date_input, "%Y-%m-%d").month
    elif isinstance(date_input, datetime):
        month = date_input.month
    elif isinstance(date_input, (int, np.integer)):
        month = int(date_input)
    else:
        month = date_input.month

    return SEASON_MAP.get(month, "Unknown")


def get_season_emoji(season: str) -> str:
    """Get emoji for a season."""
    emojis = {
        "Summer": "☀️",
        "Monsoon": "🌧️",
        "Post-Monsoon": "🍂",
        "Winter": "❄️",
    }
    return emojis.get(season, "🌍")


def recommend_stock(
    predicted_demand: float,
    season: str,
    product_category: str,
) -> dict:
    """Calculate recommended stock level with safety buffer.

    Returns dict with:
        - predicted_demand
        - buffer_pct
        - buffer_units
        - recommended_stock
        - seasonal_factor
    """
    # Base buffer from season
    base_buffer = STOCK_BUFFER.get(season, 0.15)

    # Extra buffer if there's a seasonal demand spike for this category
    seasonal_factor = SEASONAL_MULTIPLIERS.get((season, product_category), 1.0)
    if seasonal_factor > 1.3:
        base_buffer += 0.05  # extra 5% for high-spike combos

    buffer_units = int(predicted_demand * base_buffer)
    recommended = int(predicted_demand + buffer_units)

    return {
        "predicted_demand": int(predicted_demand),
        "buffer_pct": round(base_buffer * 100, 1),
        "buffer_units": buffer_units,
        "recommended_stock": recommended,
        "seasonal_factor": round(seasonal_factor, 2),
    }


def generate_alert(
    current_stock: int,
    recommended_stock: int,
    product_name: str,
) -> dict | None:
    """Generate stock alert if levels are concerning.

    Returns:
        dict with alert info, or None if stock is OK.
    """
    if recommended_stock == 0:
        return None

    ratio = current_stock / recommended_stock

    if ratio < LOW_STOCK_THRESHOLD:
        deficit = recommended_stock - current_stock
        return {
            "type": "LOW_STOCK",
            "severity": "critical" if ratio < 0.5 else "warning",
            "product": product_name,
            "current_stock": current_stock,
            "recommended_stock": recommended_stock,
            "deficit": deficit,
            "message": f"⚠️ {product_name}: Stock is {ratio:.0%} of recommended. Order {deficit} more units.",
            "icon": "🔴" if ratio < 0.5 else "🟠",
        }
    elif ratio > OVERSTOCK_THRESHOLD:
        excess = current_stock - recommended_stock
        return {
            "type": "OVERSTOCK",
            "severity": "info",
            "product": product_name,
            "current_stock": current_stock,
            "recommended_stock": recommended_stock,
            "excess": excess,
            "message": f"📦 {product_name}: Overstocked by {excess} units. Risk of expiry.",
            "icon": "🟡",
        }

    return None


def generate_all_alerts(
    stock_levels: dict[str, int],
    recommendations: dict[str, dict],
) -> list[dict]:
    """Generate alerts for all products.

    Args:
        stock_levels: {product_id: current_stock_quantity}
        recommendations: {product_id: recommend_stock() output}

    Returns:
        List of alert dicts, sorted by severity.
    """
    alerts = []
    severity_order = {"critical": 0, "warning": 1, "info": 2}

    for pid, rec in recommendations.items():
        current = stock_levels.get(pid, 0)
        product_name = PRODUCTS.get(pid, {}).get("name", pid)
        alert = generate_alert(current, rec["recommended_stock"], product_name)
        if alert:
            alert["product_id"] = pid
            alerts.append(alert)

    alerts.sort(key=lambda a: severity_order.get(a.get("severity", "info"), 3))
    return alerts


def simulate_current_stock(recommendations: dict, seed: int = 42) -> dict:
    """Simulate current stock levels for demo purposes.

    Creates a mix of well-stocked, low-stock, and overstocked products.
    """
    rng = np.random.RandomState(seed)
    stock = {}
    for pid, rec in recommendations.items():
        rec_stock = rec["recommended_stock"]
        # Randomly assign: 30% low, 20% overstock, 50% normal
        roll = rng.random()
        if roll < 0.3:
            stock[pid] = int(rec_stock * rng.uniform(0.3, 0.75))
        elif roll < 0.5:
            stock[pid] = int(rec_stock * rng.uniform(1.5, 2.0))
        else:
            stock[pid] = int(rec_stock * rng.uniform(0.85, 1.15))
    return stock
