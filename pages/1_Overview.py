"""
Page 1: Store Overview
KPI summary, all-product stock table, alert panel, business KPIs.
"""
import os, sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import PRODUCTS, SEASON_COLORS, FAST_MOVING_THRESHOLD, SLOW_MOVING_THRESHOLD
from src.data_preprocessing import load_and_preprocess, prepare_product_data
from src.stock_recommender import get_season, get_season_emoji, recommend_stock, generate_all_alerts, classify_product
from src.inventory import get_all_stock_levels

st.set_page_config(page_title="PharmaCast — Overview", page_icon="🏠", layout="wide")
# Shared CSS injected below

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:linear-gradient(160deg,#0d1117 0%,#0a1628 40%,#071a20 100%);}
.kpi-card{background:linear-gradient(135deg,rgba(13,25,40,.95),rgba(10,30,45,.95));border:1px solid rgba(20,184,166,.25);border-radius:16px;padding:20px 24px;text-align:center;backdrop-filter:blur(12px);box-shadow:0 8px 32px rgba(0,0,0,.4),inset 0 1px 0 rgba(20,184,166,.1);transition:transform .2s,box-shadow .2s,border-color .2s;}
.kpi-card:hover{transform:translateY(-4px);box-shadow:0 16px 48px rgba(20,184,166,.2);border-color:rgba(20,184,166,.5);}
.kpi-value{font-size:2.2rem;font-weight:800;background:linear-gradient(135deg,#14b8a6,#06b6d4,#10b981);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:4px 0;}
.kpi-label{font-size:.78rem;color:#64748b;text-transform:uppercase;letter-spacing:1.5px;font-weight:600;}
.kpi-icon{font-size:1.8rem;margin-bottom:6px;}
.section-header{font-size:1.15rem;font-weight:700;color:#e2e8f0;padding:8px 0 14px;border-bottom:2px solid rgba(20,184,166,.3);margin-bottom:18px;letter-spacing:.3px;}
.alert-critical{background:linear-gradient(135deg,rgba(220,38,38,.15),rgba(220,38,38,.05));border-left:4px solid #ef4444;border-radius:10px;padding:12px 16px;margin:8px 0;color:#fca5a5;}
.alert-warning{background:linear-gradient(135deg,rgba(245,158,11,.15),rgba(245,158,11,.05));border-left:4px solid #f59e0b;border-radius:10px;padding:12px 16px;margin:8px 0;color:#fcd34d;}
.alert-info{background:linear-gradient(135deg,rgba(20,184,166,.15),rgba(20,184,166,.05));border-left:4px solid #14b8a6;border-radius:10px;padding:12px 16px;margin:8px 0;color:#5eead4;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0a1628 0%,#0d1117 100%);border-right:1px solid rgba(20,184,166,.15);}
#MainMenu{visibility:hidden;}footer{visibility:hidden;}header{visibility:hidden;}
</style>""", unsafe_allow_html=True)

# ─── Data ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    from config import DATA_FILE
    if not os.path.exists(DATA_FILE):
        st.error("❌ Data not found. Run `python generate_dataset.py` first.")
        st.stop()
    return load_and_preprocess(DATA_FILE)

df = load_data()

current_month  = datetime.now().month
current_season = get_season(current_month)
season_emoji   = get_season_emoji(current_season)
HORIZON        = 30  # overview always uses 30-day horizon

# Generate forecasts + recommendations for all products
@st.cache_data
def compute_overview(horizon):
    all_recs, all_forecasts = {}, {}
    for pid, pinfo in PRODUCTS.items():
        pdata = prepare_product_data(df, pid)
        qty   = pdata["quantity_sold"].values
        last30 = qty[-30:]
        mean_v = np.mean(last30); trend = np.polyfit(range(len(last30)), last30, 1)[0]
        preds  = [max(1, int(mean_v + trend * d)) for d in range(horizon)]
        total  = int(np.sum(preds))
        all_forecasts[pid] = total
        lead = pinfo.get("lead_time_days", 5)
        avg_daily = float(np.mean(qty[-30:]))
        all_recs[pid] = recommend_stock(total, current_season, pinfo["category"],
                                         avg_daily_demand=avg_daily, lead_time_days=lead)
    return all_recs, all_forecasts

all_recs, all_forecasts = compute_overview(HORIZON)

# Real stock levels
stock_levels = get_all_stock_levels()
all_alerts   = generate_all_alerts(stock_levels, all_recs)

# ─── Header ───────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='background:linear-gradient(135deg,#14b8a6,#06b6d4);-webkit-background-clip:text;"
    "-webkit-text-fill-color:transparent;font-size:2.2rem;font-weight:800;margin-bottom:0;'>"
    "🏠 Store Overview</h1>"
    "<p style='color:#64748b;margin-top:0;'>Live inventory health for MediCare Pharmacy</p>",
    unsafe_allow_html=True,
)

# ─── KPI Row ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📈 Key Performance Indicators</div>', unsafe_allow_html=True)
k1, k2, k3, k4, k5 = st.columns(5)

total_forecast_all = sum(all_forecasts.values())
low_count      = sum(1 for a in all_alerts if a.get("severity") in ("critical", "warning"))
overstock_count= sum(1 for a in all_alerts if a.get("type") == "OVERSTOCK")
avg_seasonal   = round(np.mean([r["seasonal_factor"] for r in all_recs.values()]), 2)

for col, (icon, val, label) in zip(
    [k1, k2, k3, k4, k5],
    [
        ("💊", len(PRODUCTS), "Products Tracked"),
        ("📦", f"{total_forecast_all:,}", f"Forecast Units ({HORIZON}d)"),
        ("🚨", low_count, "Low Stock Alerts"),
        ("📦", overstock_count, "Overstock Alerts"),
        (season_emoji, current_season, "Current Season"),
    ],
):
    with col:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-icon">{icon}</div>
          <div class="kpi-value">{val}</div>
          <div class="kpi-label">{label}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("")

# ─── Stock Table ──────────────────────────────────────────────────────
st.markdown('<div class="section-header">📋 All-Product Stock Status</div>', unsafe_allow_html=True)

table_rows = []
for pid, pinfo in PRODUCTS.items():
    rec    = all_recs[pid]
    cur    = stock_levels.get(pid, 0)
    ratio  = cur / rec["recommended_stock"] if rec["recommended_stock"] > 0 else 1.0
    avg_d  = rec["avg_daily_demand"]
    mover  = classify_product(avg_d)

    alert_for = next((a for a in all_alerts if a.get("product_id") == pid), None)
    if alert_for:
        status = alert_for["label"]
    elif ratio >= 0.8:
        status = "✅ Healthy"
    else:
        status = "✅ Healthy"

    table_rows.append({
        "Product":             pinfo["name"],
        "Category":            pinfo["category"].title(),
        "Avg Daily Demand":    f"{avg_d:.0f}",
        "30d Forecast":        f"{rec['predicted_demand']:,}",
        "Safety Stock":        f"{rec['safety_stock']:,}",
        "Reorder Point":       f"{rec['reorder_point']:,}",
        "Recommended":         f"{rec['recommended_stock']:,}",
        "Current Stock":       f"{cur:,}",
        "Buffer %":            f"{rec['buffer_pct']}%",
        "Seasonal ×":          f"{rec['seasonal_factor']}",
        "Turnover":            mover,
        "Status":              status,
    })

st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True, height=460)
st.markdown("")

# ─── Alert Panel ─────────────────────────────────────────────────────
st.markdown('<div class="section-header">⚠️ Stock Alerts</div>', unsafe_allow_html=True)

if not all_alerts:
    st.success("✅ All stock levels are within acceptable range!")
else:
    alert_cols = st.columns(min(len(all_alerts), 3))
    for idx, alert in enumerate(all_alerts):
        col       = alert_cols[idx % 3]
        css_class = f"alert-{alert['severity']}"
        with col:
            extra = f"Deficit: {alert.get('deficit', alert.get('excess', 'N/A')):,}" if isinstance(alert.get('deficit', alert.get('excess')), int) else ""
            st.markdown(
                f'<div class="{css_class}">'
                f"<strong>{alert['icon']} {alert['product']}</strong><br>"
                f"<small>{alert['message']}</small><br>"
                f"<small>Current: {alert['current_stock']:,} | Recommended: {alert['recommended_stock']:,}"
                f"{ ' | ' + extra if extra else ''}</small>"
                f"</div>",
                unsafe_allow_html=True,
            )

st.markdown("")

# ─── Business KPIs ───────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Business KPIs</div>', unsafe_allow_html=True)
b1, b2, b3 = st.columns(3)

total_recommended = sum(r["recommended_stock"] for r in all_recs.values())
total_current     = sum(stock_levels.values())
stockout_risk_pct = round(low_count / len(PRODUCTS) * 100, 1)
overstock_pct     = round(overstock_count / len(PRODUCTS) * 100, 1)
stock_coverage    = round(total_current / max(1, total_forecast_all) * HORIZON, 1)

with b1:
    bkpis = [
        ("📉 Stock-out Risk", f"{stockout_risk_pct}% of products"),
        ("📦 Overstock Risk",  f"{overstock_pct}% of products"),
        ("🔄 Stock Coverage",  f"{stock_coverage} days avg"),
    ]
    for label, val in bkpis:
        st.markdown(f"""<div style="display:flex;justify-content:space-between;
            padding:8px 14px;margin:5px 0;border-radius:8px;
            background:rgba(255,255,255,0.03);">
            <span style="color:#8b8fa3;font-size:.88rem;">{label}</span>
            <span style="color:#e2e8f0;font-weight:600;font-size:.88rem;">{val}</span>
            </div>""", unsafe_allow_html=True)

with b2:
    bkpis2 = [
        ("🏪 Total Recommended", f"{total_recommended:,} units"),
        ("📋 Current Inventory",  f"{total_current:,} units"),
        ("⚡ Avg Seasonal Uplift", f"{avg_seasonal}×"),
    ]
    for label, val in bkpis2:
        st.markdown(f"""<div style="display:flex;justify-content:space-between;
            padding:8px 14px;margin:5px 0;border-radius:8px;
            background:rgba(255,255,255,0.03);">
            <span style="color:#8b8fa3;font-size:.88rem;">{label}</span>
            <span style="color:#e2e8f0;font-weight:600;font-size:.88rem;">{val}</span>
            </div>""", unsafe_allow_html=True)

with b3:
    fast_movers = [p["name"] for p in PRODUCTS.values()
                   if p["base_demand"] >= FAST_MOVING_THRESHOLD]
    slow_movers = [p["name"] for p in PRODUCTS.values()
                   if p["base_demand"] <= SLOW_MOVING_THRESHOLD]
    st.markdown(f"**🚀 Fast-Moving ({len(fast_movers)}):**")
    for name in fast_movers:
        st.markdown(f"<span style='color:#34d399;font-size:.85rem;'>● {name}</span>", unsafe_allow_html=True)
    st.markdown(f"**🐢 Slow-Moving ({len(slow_movers)}):**")
    for name in slow_movers:
        st.markdown(f"<span style='color:#f87171;font-size:.85rem;'>● {name}</span>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center;color:#555;font-size:.75rem;padding:10px;'>"
            "💊 PharmaCast v2.0 | Use the sidebar to navigate between pages</div>",
            unsafe_allow_html=True)
