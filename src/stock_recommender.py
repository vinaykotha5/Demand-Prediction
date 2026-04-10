"""
Stock recommendation engine and alert system — v2.

Uses a proper inventory formula:
    Safety Stock     = avg_daily_demand × lead_time × SAFETY_STOCK_FACTOR
    Reorder Point    = avg_daily_demand × lead_time + safety_stock
    Recommended Stock = forecasted_demand + safety_stock

Alert categories (6 types):
    Critical       — stock < 50% of recommended
    Warning        — stock < 80% of recommended (Restock Soon)
    Overstock      — stock > 130% of recommended
    Fast-Moving    — high turnover product label
    Slow-Moving    — below-average turnover product label
    Seasonal Risk  — seasonal factor > 1.5, stock borderline
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
    CRITICAL_THRESHOLD,
    PRODUCTS,
    SAFETY_STOCK_FACTOR,
    DEFAULT_LEAD_TIME,
    FAST_MOVING_THRESHOLD,
    SLOW_MOVING_THRESHOLD,
)


def get_season(date_input) -> str:
    """Determine Indian season from a date or month number."""
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
    """Return emoji for a season."""
    return {
        "Summer":       "☀️",
        "Monsoon":      "🌧️",
        "Post-Monsoon": "🍂",
        "Winter":       "❄️",
    }.get(season, "🌍")


def compute_safety_stock(
    avg_daily_demand: float,
    lead_time_days: int,
    factor: float = None,
) -> int:
    """
    Safety Stock = avg_daily_demand × lead_time_days × factor

    Args:
        avg_daily_demand: historical average units sold per day
        lead_time_days: supplier lead time
        factor: safety multiplier (default from config)
    """
    if factor is None:
        factor = SAFETY_STOCK_FACTOR
    return max(1, int(avg_daily_demand * lead_time_days * factor))


def compute_reorder_point(
    avg_daily_demand: float,
    lead_time_days: int,
    safety_stock: int = None,
) -> int:
    """
    Reorder Point = avg_daily_demand × lead_time_days + safety_stock
    """
    if safety_stock is None:
        safety_stock = compute_safety_stock(avg_daily_demand, lead_time_days)
    return int(avg_daily_demand * lead_time_days) + safety_stock


def recommend_stock(
    predicted_demand: float,
    season: str,
    product_category: str,
    avg_daily_demand: float = None,
    lead_time_days: int = None,
) -> dict:
    """
    Calculate recommended stock with proper safety stock formula.

    Formula:
        safety_stock      = avg_daily × lead_time × SAFETY_STOCK_FACTOR
        reorder_point     = avg_daily × lead_time + safety_stock
        recommended_stock = forecasted_demand + safety_stock

    Returns dict with all recommendation fields.
    """
    if avg_daily_demand is None:
        avg_daily_demand = predicted_demand / 30  # estimate from 30-day forecast
    if lead_time_days is None:
        lead_time_days = DEFAULT_LEAD_TIME

    # Season buffer
    base_buffer = STOCK_BUFFER.get(season, 0.15)

    # Seasonal factor
    seasonal_factor = SEASONAL_MULTIPLIERS.get((season, product_category), 1.0)
    if seasonal_factor > 1.3:
        base_buffer += 0.05  # extra 5% for high-spike combos

    safety_stock  = compute_safety_stock(avg_daily_demand, lead_time_days)
    reorder_point = compute_reorder_point(avg_daily_demand, lead_time_days, safety_stock)

    # Recommended = forecast + safety buffer (both formula-based and season %)
    buffer_units      = max(safety_stock, int(predicted_demand * base_buffer))
    recommended_stock = int(predicted_demand) + buffer_units

    return {
        "predicted_demand":  int(predicted_demand),
        "avg_daily_demand":  round(avg_daily_demand, 1),
        "lead_time_days":    lead_time_days,
        "safety_stock":      safety_stock,
        "reorder_point":     reorder_point,
        "buffer_pct":        round(base_buffer * 100, 1),
        "buffer_units":      buffer_units,
        "recommended_stock": recommended_stock,
        "seasonal_factor":   round(seasonal_factor, 2),
    }


def classify_product(avg_daily_demand: float) -> str:
    """Classify a product as Fast-Moving, Slow-Moving, or Normal."""
    if avg_daily_demand >= FAST_MOVING_THRESHOLD:
        return "Fast-Moving"
    if avg_daily_demand <= SLOW_MOVING_THRESHOLD:
        return "Slow-Moving"
    return "Normal"


def generate_alert(
    current_stock: int,
    recommended_stock: int,
    reorder_point: int,
    product_name: str,
    avg_daily_demand: float = 0,
    seasonal_factor: float = 1.0,
) -> dict | None:
    """
    Generate a stock alert with 6 possible categories.

    Returns:
        dict with alert info, or None if stock is healthy.
    """
    if recommended_stock == 0:
        return None

    ratio = current_stock / recommended_stock

    # Critical — below 50%
    if ratio < CRITICAL_THRESHOLD:
        deficit = recommended_stock - current_stock
        return {
            "type": "CRITICAL",
            "severity": "critical",
            "label":  "🔴 Critical",
            "product": product_name,
            "current_stock": current_stock,
            "recommended_stock": recommended_stock,
            "reorder_point": reorder_point,
            "deficit": deficit,
            "message": f"Stock critically low — only {ratio:.0%} of recommended. Order {deficit:,} units urgently.",
            "icon": "🔴",
        }

    # Seasonal Risk — seasonal factor > 1.5, stock borderline
    if seasonal_factor > 1.5 and ratio < 0.9:
        deficit = recommended_stock - current_stock
        return {
            "type": "SEASONAL_RISK",
            "severity": "warning",
            "label": "🌡️ Seasonal Risk",
            "product": product_name,
            "current_stock": current_stock,
            "recommended_stock": recommended_stock,
            "reorder_point": reorder_point,
            "deficit": deficit,
            "message": f"High-season item at {ratio:.0%} stock (demand ×{seasonal_factor}). Consider early restocking.",
            "icon": "🌡️",
        }

    # Restock Soon — below reorder point
    if current_stock <= reorder_point:
        deficit = recommended_stock - current_stock
        return {
            "type": "RESTOCK_SOON",
            "severity": "warning",
            "label": "🟠 Restock Soon",
            "product": product_name,
            "current_stock": current_stock,
            "recommended_stock": recommended_stock,
            "reorder_point": reorder_point,
            "deficit": deficit,
            "message": f"Stock below reorder point ({reorder_point:,}). Place order for {deficit:,} units.",
            "icon": "🟠",
        }

    # Low Stock — below 80%
    if ratio < LOW_STOCK_THRESHOLD:
        deficit = recommended_stock - current_stock
        return {
            "type": "LOW_STOCK",
            "severity": "warning",
            "label": "🟡 Low Stock",
            "product": product_name,
            "current_stock": current_stock,
            "recommended_stock": recommended_stock,
            "reorder_point": reorder_point,
            "deficit": deficit,
            "message": f"Stock at {ratio:.0%} of recommended. Monitor closely.",
            "icon": "🟡",
        }

    # Overstock — above 130%
    if ratio > OVERSTOCK_THRESHOLD:
        excess = current_stock - recommended_stock
        return {
            "type": "OVERSTOCK",
            "severity": "info",
            "label": "📦 Overstock",
            "product": product_name,
            "current_stock": current_stock,
            "recommended_stock": recommended_stock,
            "reorder_point": reorder_point,
            "excess": excess,
            "message": f"Overstocked by {excess:,} units. Risk of expiry — delay next order.",
            "icon": "📦",
        }

    return None  # Healthy ✅


def generate_all_alerts(
    stock_levels: dict,
    recommendations: dict,
) -> list[dict]:
    """
    Generate alerts for all products, sorted by severity.

    Args:
        stock_levels   : {product_id: current_stock}
        recommendations: {product_id: recommend_stock() output}
    """
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    alerts = []

    for pid, rec in recommendations.items():
        current      = stock_levels.get(pid, 0)
        product_name = PRODUCTS.get(pid, {}).get("name", pid)
        alert = generate_alert(
            current_stock=current,
            recommended_stock=rec["recommended_stock"],
            reorder_point=rec.get("reorder_point", 0),
            product_name=product_name,
            avg_daily_demand=rec.get("avg_daily_demand", 0),
            seasonal_factor=rec.get("seasonal_factor", 1.0),
        )
        if alert:
            alert["product_id"] = pid
            alerts.append(alert)

    alerts.sort(key=lambda a: severity_order.get(a.get("severity", "info"), 3))
    return alerts


def simulate_current_stock(recommendations: dict, seed: int = 42) -> dict:
    """
    Simulate current stock levels for demo / fallback when inventory.json
    is not yet initialised.  Creates a realistic mix of stock situations.
    """
    rng = np.random.RandomState(seed)
    stock = {}
    for pid, rec in recommendations.items():
        rec_stock = rec["recommended_stock"]
        roll = rng.random()
        if roll < 0.3:
            stock[pid] = int(rec_stock * rng.uniform(0.3, 0.75))
        elif roll < 0.5:
            stock[pid] = int(rec_stock * rng.uniform(1.5, 2.0))
        else:
            stock[pid] = int(rec_stock * rng.uniform(0.85, 1.15))
    return stock
