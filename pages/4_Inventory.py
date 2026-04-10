"""
Page 4: Inventory Manager
Edit current stock levels, view safety stock, reorder points, and alert status.
All changes persist to data/inventory.json.
"""
import os, sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import PRODUCTS, FORECAST_HORIZONS
from src.data_preprocessing import load_and_preprocess, prepare_product_data
from src.stock_recommender import (
    get_season, get_season_emoji, recommend_stock,
    generate_alert, classify_product,
)
from src.inventory import load_inventory, bulk_update, reset_to_defaults

st.set_page_config(page_title="PharmaCast — Inventory", page_icon="📦", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:linear-gradient(135deg,#0f0c29 0%,#1a1a2e 50%,#16213e 100%);}
.kpi-card{background:linear-gradient(135deg,rgba(30,33,48,.92),rgba(44,47,68,.92));border:1px solid rgba(255,255,255,.08);border-radius:16px;padding:20px 24px;text-align:center;backdrop-filter:blur(10px);box-shadow:0 8px 32px rgba(0,0,0,.3);}
.kpi-value{font-size:2rem;font-weight:800;background:linear-gradient(135deg,#667eea,#764ba2);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:4px 0;}
.kpi-label{font-size:.8rem;color:#8b8fa3;text-transform:uppercase;letter-spacing:1.2px;font-weight:500;}
.kpi-icon{font-size:1.8rem;margin-bottom:4px;}
.section-header{font-size:1.25rem;font-weight:700;color:#e2e8f0;padding:8px 0 14px;border-bottom:2px solid rgba(102,126,234,.35);margin-bottom:18px;}
.alert-critical{background:linear-gradient(135deg,rgba(220,38,38,.18),rgba(220,38,38,.06));border-left:4px solid #dc2626;border-radius:8px;padding:10px 14px;margin:6px 0;color:#fca5a5;}
.alert-warning{background:linear-gradient(135deg,rgba(245,158,11,.18),rgba(245,158,11,.06));border-left:4px solid #f59e0b;border-radius:8px;padding:10px 14px;margin:6px 0;color:#fcd34d;}
.alert-info{background:linear-gradient(135deg,rgba(59,130,246,.18),rgba(59,130,246,.06));border-left:4px solid #3b82f6;border-radius:8px;padding:10px 14px;margin:6px 0;color:#93c5fd;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#1a1a2e 0%,#0f0c29 100%);}
#MainMenu{visibility:hidden;}footer{visibility:hidden;}header{visibility:hidden;}
</style>""", unsafe_allow_html=True)

st.markdown(
    "<h1 style='background:linear-gradient(135deg,#667eea,#764ba2);-webkit-background-clip:text;"
    "-webkit-text-fill-color:transparent;font-size:2.2rem;font-weight:800;margin-bottom:0;'>"
    "📦 Inventory Manager</h1>"
    "<p style='color:#8b8fa3;margin-top:0;'>Update current stock levels and view reorder recommendations. Changes are saved automatically.</p>",
    unsafe_allow_html=True,
)

# ─── Load data ────────────────────────────────────────────────────────
@st.cache_data
def load_df():
    from config import DATA_FILE
    return load_and_preprocess(DATA_FILE)

df             = load_df()
current_month  = datetime.now().month
current_season = get_season(current_month)
season_emoji   = get_season_emoji(current_season)
horizon        = 30

# Load persisted inventory
inventory = load_inventory()

# Compute recommendations for all products
@st.cache_data
def compute_recs(season, h):
    recs = {}
    for pid, pinfo in PRODUCTS.items():
        pdata  = prepare_product_data(df, pid)
        qty    = pdata["quantity_sold"].values
        last30 = qty[-30:]
        total  = int(np.sum([max(1, int(np.mean(last30) + np.polyfit(range(30), last30, 1)[0] * d)) for d in range(h)]))
        avg_d  = float(np.mean(qty[-30:]))
        lead   = pinfo.get("lead_time_days", 5)
        recs[pid] = recommend_stock(total, season, pinfo["category"],
                                     avg_daily_demand=avg_d, lead_time_days=lead)
    return recs

all_recs = compute_recs(current_season, horizon)

# ─── Inventory Formula Info Box ───────────────────────────────────────
st.markdown('<div class="section-header">📐 Inventory Formulas Used</div>', unsafe_allow_html=True)
fi1, fi2, fi3 = st.columns(3)
with fi1:
    st.markdown("""
    <div style="background:rgba(102,126,234,0.1);border-radius:12px;padding:16px;border:1px solid rgba(102,126,234,0.2);">
    <b style="color:#a5b4fc;">🛡️ Safety Stock</b><br>
    <code style="color:#e2e8f0;font-size:.95rem;">Avg Daily × Lead Time × 0.5</code><br>
    <small style="color:#64748b;">Buffer for demand variability & supply delays.</small>
    </div>""", unsafe_allow_html=True)
with fi2:
    st.markdown("""
    <div style="background:rgba(102,126,234,0.1);border-radius:12px;padding:16px;border:1px solid rgba(102,126,234,0.2);">
    <b style="color:#a5b4fc;">🔄 Reorder Point</b><br>
    <code style="color:#e2e8f0;font-size:.95rem;">Avg Daily × Lead Time + Safety Stock</code><br>
    <small style="color:#64748b;">Place order when stock falls to this level.</small>
    </div>""", unsafe_allow_html=True)
with fi3:
    st.markdown("""
    <div style="background:rgba(102,126,234,0.1);border-radius:12px;padding:16px;border:1px solid rgba(102,126,234,0.2);">
    <b style="color:#a5b4fc;">🏪 Recommended Stock</b><br>
    <code style="color:#e2e8f0;font-size:.95rem;">Forecast + max(Safety Stock, Forecast × Buffer%)</code><br>
    <small style="color:#64748b;">Ideal on-hand stock for the forecast period.</small>
    </div>""", unsafe_allow_html=True)
st.markdown("")

# ─── Stock Editor ─────────────────────────────────────────────────────
st.markdown('<div class="section-header">✏️ Update Current Stock Levels</div>', unsafe_allow_html=True)
st.markdown(
    "<div style='background:rgba(102,126,234,0.08);border-radius:8px;padding:10px 14px;"
    "color:#94a3b8;font-size:.85rem;margin-bottom:16px;border:1px solid rgba(102,126,234,0.15);'>"
    "💡 Enter current physical stock counts below. Click <b>Save Inventory</b> to persist changes."
    "</div>",
    unsafe_allow_html=True,
)

new_stock = {}
alert_messages = []

# Build 3-column grid of stock editors
pid_list = list(PRODUCTS.keys())
rows_of_3 = [pid_list[i:i+3] for i in range(0, len(pid_list), 3)]

for row_pids in rows_of_3:
    cols = st.columns(3)
    for col, pid in zip(cols, row_pids):
        pinfo  = PRODUCTS[pid]
        rec    = all_recs[pid]
        cur    = inventory.get(pid, {}).get("current_stock", 0)
        mover  = classify_product(rec["avg_daily_demand"])
        ratio  = cur / rec["recommended_stock"] if rec["recommended_stock"] > 0 else 1.0

        # Status colour
        if ratio < 0.5:
            status_color = "#f87171"; status_text = "🔴 Critical"
        elif ratio < 0.8:
            status_color = "#fcd34d"; status_text = "🟠 Low"
        elif ratio > 1.3:
            status_color = "#60a5fa"; status_text = "📦 Overstock"
        else:
            status_color = "#34d399"; status_text = "✅ OK"

        with col:
            with st.container():
                st.markdown(
                    f"<div style='background:rgba(30,33,48,0.8);border:1px solid rgba(255,255,255,0.07);"
                    f"border-radius:12px;padding:14px;margin-bottom:12px;'>"
                    f"<div style='font-weight:700;color:#e2e8f0;font-size:.95rem;margin-bottom:4px;'>{pinfo['name']}</div>"
                    f"<div style='font-size:.78rem;color:#64748b;margin-bottom:8px;'>{pinfo['category'].title()} · {mover}</div>"
                    f"<div style='display:flex;justify-content:space-between;font-size:.8rem;margin-bottom:6px;'>"
                    f"<span style='color:#8b8fa3;'>Reorder Point</span>"
                    f"<span style='color:#a5b4fc;font-weight:600;'>{rec['reorder_point']:,}</span></div>"
                    f"<div style='display:flex;justify-content:space-between;font-size:.8rem;margin-bottom:6px;'>"
                    f"<span style='color:#8b8fa3;'>Safety Stock</span>"
                    f"<span style='color:#a5b4fc;font-weight:600;'>{rec['safety_stock']:,}</span></div>"
                    f"<div style='display:flex;justify-content:space-between;font-size:.8rem;margin-bottom:10px;'>"
                    f"<span style='color:#8b8fa3;'>Recommended</span>"
                    f"<span style='color:#e2e8f0;font-weight:600;'>{rec['recommended_stock']:,}</span></div>"
                    f"<span style='font-size:.78rem;font-weight:600;color:{status_color};'>{status_text}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                new_val = st.number_input(
                    f"Current Stock — {pid}",
                    min_value=0,
                    value=cur,
                    step=10,
                    key=f"stock_{pid}",
                    label_visibility="collapsed",
                )
                new_stock[pid] = new_val

                # Queue alert
                alert = generate_alert(
                    current_stock=new_val,
                    recommended_stock=rec["recommended_stock"],
                    reorder_point=rec["reorder_point"],
                    product_name=pinfo["name"],
                    avg_daily_demand=rec["avg_daily_demand"],
                    seasonal_factor=rec["seasonal_factor"],
                )
                if alert:
                    alert_messages.append(alert)

st.markdown("")

# ─── Save Button + Reset ──────────────────────────────────────────────
sv1, sv2 = st.columns([3, 1])
with sv1:
    if st.button("💾 Save Inventory", type="primary", use_container_width=True):
        bulk_update(new_stock)
        st.success("✅ Inventory saved successfully!")
        st.cache_data.clear()
with sv2:
    if st.button("🔁 Reset to Defaults", use_container_width=True):
        reset_to_defaults()
        st.warning("Inventory reset to default 30-day buffer levels.")
        st.rerun()

st.markdown("")

# ─── Active Alerts ────────────────────────────────────────────────────
st.markdown('<div class="section-header">⚠️ Current Alerts (Based on Entered Levels)</div>', unsafe_allow_html=True)

if not alert_messages:
    st.success("✅ All entered stock levels are within acceptable range!")
else:
    alert_cols = st.columns(min(len(alert_messages), 3))
    for idx, alert in enumerate(alert_messages):
        col = alert_cols[idx % 3]
        css = f"alert-{alert['severity']}"
        with col:
            st.markdown(
                f'<div class="{css}">'
                f"<strong>{alert['icon']} {alert['product']}</strong><br>"
                f"<small>{alert['message']}</small>"
                f"</div>",
                unsafe_allow_html=True,
            )

st.markdown("")

# ─── Stock vs Recommended Chart ───────────────────────────────────────
st.markdown('<div class="section-header">📊 Current Stock vs Recommended (All Products)</div>', unsafe_allow_html=True)

products_names = [PRODUCTS[pid]["name"] for pid in PRODUCTS]
cur_stocks     = [new_stock.get(pid, inventory.get(pid, {}).get("current_stock", 0)) for pid in PRODUCTS]
rec_stocks     = [all_recs[pid]["recommended_stock"] for pid in PRODUCTS]
reorder_pts    = [all_recs[pid]["reorder_point"] for pid in PRODUCTS]

fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(name="Current Stock",    x=products_names, y=cur_stocks,
                          marker_color="rgba(102,126,234,0.8)"))
fig_bar.add_trace(go.Bar(name="Recommended Stock", x=products_names, y=rec_stocks,
                          marker_color="rgba(240,147,251,0.6)"))
fig_bar.add_trace(go.Scatter(name="Reorder Point", x=products_names, y=reorder_pts,
                              mode="markers+lines",
                              marker=dict(symbol="diamond", size=10, color="#f59e0b"),
                              line=dict(color="#f59e0b", width=1.5, dash="dot")))
fig_bar.update_layout(
    barmode="group", height=420,
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=40, r=20, t=40, b=100),
    font=dict(family="Inter"), yaxis_title="Units",
    xaxis_tickangle=-30,
    legend=dict(orientation="h", y=1.05, x=1, xanchor="right"),
)
fig_bar.update_xaxes(gridcolor="rgba(255,255,255,.05)")
fig_bar.update_yaxes(gridcolor="rgba(255,255,255,.05)")
st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")
st.markdown("<div style='text-align:center;color:#555;font-size:.75rem;padding:10px;'>"
            "💊 PharmaCast v2.0 | Inventory Manager</div>", unsafe_allow_html=True)
