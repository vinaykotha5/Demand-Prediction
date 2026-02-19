"""
📊 Pharmacy Demand Prediction & Stock Analytics Dashboard
Real-time analytics dashboard powered by LSTM demand forecasting.
Run: streamlit run app.py
"""
import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    PRODUCTS,
    SEASON_MAP,
    SEASON_COLORS,
    FORECAST_HORIZONS,
    STORE_NAME,
    DATA_FILE,
    MODEL_DIR,
    SCALER_DIR,
    SEQUENCE_LENGTH,
)
from src.data_preprocessing import load_and_preprocess, prepare_product_data, create_features, scale_data
from src.stock_recommender import (
    get_season,
    get_season_emoji,
    recommend_stock,
    generate_all_alerts,
    simulate_current_stock,
)

# ─── Page Config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="PharmaCast — Demand Prediction",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark premium theme overrides */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
    }

    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, rgba(30, 33, 48, 0.9), rgba(44, 47, 68, 0.9));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 20px 24px;
        text-align: center;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.4);
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 4px 0;
    }
    .kpi-label {
        font-size: 0.85rem;
        color: #8b8fa3;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 500;
    }
    .kpi-icon {
        font-size: 1.8rem;
        margin-bottom: 4px;
    }

    /* Alert cards */
    .alert-critical {
        background: linear-gradient(135deg, rgba(220,38,38,0.15), rgba(220,38,38,0.05));
        border-left: 4px solid #dc2626;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        color: #fca5a5;
    }
    .alert-warning {
        background: linear-gradient(135deg, rgba(245,158,11,0.15), rgba(245,158,11,0.05));
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        color: #fcd34d;
    }
    .alert-info {
        background: linear-gradient(135deg, rgba(59,130,246,0.15), rgba(59,130,246,0.05));
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        color: #93c5fd;
    }

    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #e2e8f0;
        padding: 8px 0 16px 0;
        border-bottom: 2px solid rgba(102, 126, 234, 0.3);
        margin-bottom: 16px;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0f0c29 100%);
    }

    /* Hide default streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Table styling */
    .dataframe {
        border-radius: 8px !important;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(30, 33, 48, 0.8), rgba(44, 47, 68, 0.8));
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 16px;
    }
</style>
""", unsafe_allow_html=True)


# ─── Data Loading ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load and cache the preprocessed sales data."""
    if not os.path.exists(DATA_FILE):
        st.error(f"❌ Data file not found: {DATA_FILE}\n\nRun `python generate_dataset.py` first.")
        st.stop()
    return load_and_preprocess(DATA_FILE)


@st.cache_data
def get_historical_summary(df, product_id):
    """Get historical stats for a product."""
    pdf = prepare_product_data(df, product_id)
    return {
        "total_sold": int(pdf["quantity_sold"].sum()),
        "avg_daily": round(pdf["quantity_sold"].mean(), 1),
        "max_daily": int(pdf["quantity_sold"].max()),
        "min_daily": int(pdf["quantity_sold"].min()),
        "std": round(pdf["quantity_sold"].std(), 1),
        "data": pdf,
    }


def generate_forecast(product_data, horizon):
    """Generate demand forecast. Uses trained model if available, otherwise statistical forecast."""
    try:
        from src.model import load_trained_model, predict_demand
        import joblib

        product_id = product_data["product_id"].iloc[0]
        model = load_trained_model(product_id)
        scaler_path = os.path.join(SCALER_DIR, f"scaler_{product_id}.pkl")
        scaler = joblib.load(scaler_path)

        features = create_features(product_data)
        scaled, _ = scale_data(features, product_id, fit=False)
        last_seq = scaled[-SEQUENCE_LENGTH:]

        predictions = predict_demand(model, last_seq, scaler, horizon)
        return predictions, "LSTM Model"

    except Exception:
        # Fallback: statistical forecast using seasonal decomposition
        qty = product_data["quantity_sold"].values
        last_30 = qty[-30:]
        mean_val = np.mean(last_30)
        std_val = np.std(last_30)
        trend = np.polyfit(range(len(last_30)), last_30, 1)[0]

        predictions = []
        for d in range(horizon):
            pred = mean_val + trend * d + np.random.normal(0, std_val * 0.3)
            predictions.append(max(1, int(pred)))

        return np.array(predictions), "Statistical (Fallback)"


# ─── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💊 PharmaCast")
    st.markdown(f"*{STORE_NAME}*")
    st.markdown("---")

    # Product selector
    product_options = {f"{v['name']} ({k})": k for k, v in PRODUCTS.items()}
    selected_label = st.selectbox(
        "🔎 Select Product",
        options=list(product_options.keys()),
        index=0,
    )
    selected_product = product_options[selected_label]

    st.markdown("---")

    # Forecast horizon
    horizon = st.selectbox(
        "📅 Forecast Horizon",
        options=FORECAST_HORIZONS,
        format_func=lambda x: f"{x} days",
        index=1,  # default 30 days
    )

    st.markdown("---")

    # Current date/season info
    current_month = datetime.now().month
    current_season = get_season(current_month)
    season_emoji = get_season_emoji(current_season)

    st.markdown(f"### {season_emoji} Current Season")
    st.markdown(f"**{current_season}**")
    st.markdown(f"📅 {datetime.now().strftime('%B %d, %Y')}")

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#555; font-size:0.75rem;'>"
        "Powered by LSTM Neural Network<br>© 2025 PharmaCast"
        "</div>",
        unsafe_allow_html=True,
    )


# ─── Main Content ────────────────────────────────────────────────────
df = load_data()

# Title
st.markdown(
    "<h1 style='text-align:center; background: linear-gradient(135deg, #667eea, #764ba2);"
    "-webkit-background-clip:text; -webkit-text-fill-color:transparent;"
    "font-size:2.5rem; font-weight:800; margin-bottom:0;'>"
    "📊 Demand Prediction & Stock Analytics</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; color:#8b8fa3; margin-top:0;'>"
    "LSTM-powered demand forecasting for smarter inventory management</p>",
    unsafe_allow_html=True,
)
st.markdown("")

# ─── Generate Forecast ───────────────────────────────────────────────
product_info = PRODUCTS[selected_product]
summary = get_historical_summary(df, selected_product)
product_data = summary["data"]

forecast, model_type = generate_forecast(product_data, horizon)
total_forecast = int(np.sum(forecast))

# Stock recommendation
rec = recommend_stock(total_forecast, current_season, product_info["category"])
current_stock_sim = simulate_current_stock({selected_product: rec})
current_stock = current_stock_sim[selected_product]

# ─── KPI Row ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📈 Key Performance Indicators</div>', unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">📦</div>
        <div class="kpi-value">{total_forecast:,}</div>
        <div class="kpi-label">Predicted Demand ({horizon}d)</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">🏪</div>
        <div class="kpi-value">{rec['recommended_stock']:,}</div>
        <div class="kpi-label">Recommended Stock</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">{season_emoji}</div>
        <div class="kpi-value">{current_season}</div>
        <div class="kpi-label">Current Season</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    seasonal_factor_display = f"{rec['seasonal_factor']}x"
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">🔥</div>
        <div class="kpi-value">{seasonal_factor_display}</div>
        <div class="kpi-label">Seasonal Multiplier</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")

# ─── Forecast Chart ──────────────────────────────────────────────────
st.markdown('<div class="section-header">📈 Demand Forecast</div>', unsafe_allow_html=True)

# Prepare historical data (last 90 days)
hist_df = product_data.tail(90).copy()
last_date = hist_df["date"].max()

# Forecast dates
forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon, freq="D")
forecast_df = pd.DataFrame({
    "date": forecast_dates,
    "quantity": forecast,
})

# Build chart
fig_forecast = go.Figure()

# Historical line
fig_forecast.add_trace(go.Scatter(
    x=hist_df["date"],
    y=hist_df["quantity_sold"],
    mode="lines",
    name="Historical Sales",
    line=dict(color="#667eea", width=2),
    fill="tozeroy",
    fillcolor="rgba(102,126,234,0.1)",
))

# Forecast line
fig_forecast.add_trace(go.Scatter(
    x=forecast_df["date"],
    y=forecast_df["quantity"],
    mode="lines+markers",
    name=f"Forecast ({model_type})",
    line=dict(color="#f093fb", width=2.5, dash="dot"),
    marker=dict(size=4),
))

# Confidence band (±15%)
upper = forecast_df["quantity"] * 1.15
lower = forecast_df["quantity"] * 0.85

fig_forecast.add_trace(go.Scatter(
    x=pd.concat([forecast_df["date"], forecast_df["date"][::-1]]),
    y=pd.concat([upper, lower[::-1]]),
    fill="toself",
    fillcolor="rgba(240,147,251,0.1)",
    line=dict(color="rgba(0,0,0,0)"),
    name="Confidence Band (±15%)",
    showlegend=True,
))

fig_forecast.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=400,
    margin=dict(l=40, r=20, t=30, b=40),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        font=dict(size=11),
    ),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", title=""),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Quantity"),
    hovermode="x unified",
)

st.plotly_chart(fig_forecast, use_container_width=True)

# Forecast info badge
st.markdown(
    f"<div style='text-align:center; padding:8px; background:rgba(102,126,234,0.1);"
    f"border-radius:8px; color:#a5b4fc; font-size:0.85rem;'>"
    f"🧠 Model: <b>{model_type}</b> &nbsp;|&nbsp; "
    f"Avg daily forecast: <b>{int(np.mean(forecast)):,}</b> units &nbsp;|&nbsp; "
    f"Buffer: <b>{rec['buffer_pct']}%</b> ({rec['buffer_units']:,} units)"
    f"</div>",
    unsafe_allow_html=True,
)
st.markdown("")

# ─── Two-Column: Seasonal + Stock ────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.markdown('<div class="section-header">🌦️ Seasonal Demand Analysis</div>', unsafe_allow_html=True)

    # Average demand by season
    seasonal_df = product_data.groupby("season")["quantity_sold"].mean().reset_index()
    seasonal_df.columns = ["Season", "Avg Daily Demand"]
    seasonal_df["Color"] = seasonal_df["Season"].map(SEASON_COLORS)

    fig_season = px.bar(
        seasonal_df,
        x="Season",
        y="Avg Daily Demand",
        color="Season",
        color_discrete_map=SEASON_COLORS,
        text="Avg Daily Demand",
    )
    fig_season.update_traces(
        texttemplate="%{text:.0f}",
        textposition="outside",
        marker_line_width=0,
    )
    fig_season.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=350,
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=40),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title=""),
    )
    st.plotly_chart(fig_season, use_container_width=True)

with col_right:
    st.markdown('<div class="section-header">📅 Monthly Heatmap</div>', unsafe_allow_html=True)

    # Monthly heatmap
    product_data_copy = product_data.copy()
    product_data_copy["year"] = product_data_copy["date"].dt.year
    monthly_pivot = product_data_copy.pivot_table(
        values="quantity_sold",
        index="year",
        columns="month",
        aggfunc="mean",
    ).round(0)

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig_heatmap = go.Figure(data=go.Heatmap(
        z=monthly_pivot.values,
        x=month_labels,
        y=[str(y) for y in monthly_pivot.index],
        colorscale=[
            [0, "#0f0c29"],
            [0.25, "#1a1a6c"],
            [0.5, "#667eea"],
            [0.75, "#764ba2"],
            [1, "#f093fb"],
        ],
        text=monthly_pivot.values.astype(int),
        texttemplate="%{text}",
        textfont=dict(size=11, color="white"),
        hovertemplate="Year: %{y}<br>Month: %{x}<br>Avg Qty: %{z:.0f}<extra></extra>",
    ))

    fig_heatmap.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=350,
        margin=dict(l=20, r=20, t=20, b=40),
        xaxis=dict(title=""),
        yaxis=dict(title=""),
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

st.markdown("")

# ─── Stock Recommendations Table ─────────────────────────────────────
st.markdown('<div class="section-header">📦 Stock Recommendations — All Products</div>', unsafe_allow_html=True)

all_recs = {}
all_forecasts = {}

for pid, pinfo in PRODUCTS.items():
    p_summary = get_historical_summary(df, pid)
    p_forecast, _ = generate_forecast(p_summary["data"], horizon)
    total = int(np.sum(p_forecast))
    all_forecasts[pid] = total
    all_recs[pid] = recommend_stock(total, current_season, pinfo["category"])

# Simulate current stock
all_stock = simulate_current_stock(all_recs)

# Generate alerts
all_alerts = generate_all_alerts(all_stock, all_recs)

# Build table data
table_data = []
for pid, pinfo in PRODUCTS.items():
    rec_info = all_recs[pid]
    cur = all_stock[pid]
    ratio = cur / rec_info["recommended_stock"] if rec_info["recommended_stock"] > 0 else 1.0

    if ratio < 0.5:
        status = "🔴 Critical"
    elif ratio < 0.8:
        status = "🟠 Low Stock"
    elif ratio > 1.5:
        status = "🟡 Overstock"
    else:
        status = "🟢 OK"

    table_data.append({
        "Product": pinfo["name"],
        "Category": pinfo["category"].title(),
        f"Predicted ({horizon}d)": f"{rec_info['predicted_demand']:,}",
        "Buffer %": f"{rec_info['buffer_pct']}%",
        "Recommended": f"{rec_info['recommended_stock']:,}",
        "Current Stock": f"{cur:,}",
        "Status": status,
    })

stock_df = pd.DataFrame(table_data)

st.dataframe(
    stock_df,
    use_container_width=True,
    hide_index=True,
    height=460,
)

st.markdown("")

# ─── Alerts Panel ────────────────────────────────────────────────────
st.markdown('<div class="section-header">⚠️ Stock Alerts</div>', unsafe_allow_html=True)

if not all_alerts:
    st.success("✅ All stock levels are within acceptable range!")
else:
    alert_cols = st.columns(min(len(all_alerts), 3))
    for idx, alert in enumerate(all_alerts):
        col = alert_cols[idx % 3]
        css_class = f"alert-{alert['severity']}"
        with col:
            st.markdown(
                f'<div class="{css_class}">'
                f"<strong>{alert['icon']} {alert['product']}</strong><br>"
                f"<small>{alert['message']}</small><br>"
                f"<small>Current: {alert['current_stock']:,} | "
                f"Recommended: {alert['recommended_stock']:,}</small>"
                f"</div>",
                unsafe_allow_html=True,
            )

st.markdown("")

# ─── Product Deep-Dive ───────────────────────────────────────────────
st.markdown(
    f'<div class="section-header">🔍 Product Deep-Dive — {product_info["name"]}</div>',
    unsafe_allow_html=True,
)

dd_col1, dd_col2 = st.columns([2, 1])

with dd_col1:
    # Daily trend with 7-day moving average
    trend_df = product_data.copy()
    trend_df["MA_7"] = trend_df["quantity_sold"].rolling(7).mean()
    trend_df["MA_30"] = trend_df["quantity_sold"].rolling(30).mean()

    fig_trend = go.Figure()

    fig_trend.add_trace(go.Scatter(
        x=trend_df["date"],
        y=trend_df["quantity_sold"],
        mode="lines",
        name="Daily Sales",
        line=dict(color="rgba(102,126,234,0.3)", width=1),
    ))

    fig_trend.add_trace(go.Scatter(
        x=trend_df["date"],
        y=trend_df["MA_7"],
        mode="lines",
        name="7-day MA",
        line=dict(color="#667eea", width=2.5),
    ))

    fig_trend.add_trace(go.Scatter(
        x=trend_df["date"],
        y=trend_df["MA_30"],
        mode="lines",
        name="30-day MA",
        line=dict(color="#f093fb", width=2.5),
    ))

    fig_trend.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=350,
        margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", title=""),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Quantity"),
        hovermode="x unified",
    )
    st.plotly_chart(fig_trend, use_container_width=True)

with dd_col2:
    st.markdown("#### 📊 Historical Stats")

    stat_items = [
        ("📈 Avg Daily", f"{summary['avg_daily']:,}"),
        ("🔝 Max Daily", f"{summary['max_daily']:,}"),
        ("🔻 Min Daily", f"{summary['min_daily']:,}"),
        ("📊 Std Dev", f"{summary['std']:,}"),
        ("📦 Total Sold", f"{summary['total_sold']:,}"),
        ("💰 Unit Price", f"₹{product_info['unit_price']:.0f}"),
        ("🏷️ Category", product_info["category"].title()),
        ("🔥 Season Factor", f"{rec['seasonal_factor']}x"),
    ]

    for label, value in stat_items:
        st.markdown(
            f"<div style='display:flex; justify-content:space-between; "
            f"padding:6px 12px; margin:4px 0; border-radius:6px; "
            f"background:rgba(255,255,255,0.03);'>"
            f"<span style='color:#8b8fa3;'>{label}</span>"
            f"<span style='color:#e2e8f0; font-weight:600;'>{value}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

st.markdown("")

# ─── Day-of-Week Analysis ────────────────────────────────────────────
st.markdown('<div class="section-header">📅 Day-of-Week Demand Pattern</div>', unsafe_allow_html=True)

dow_df = product_data.groupby("day_of_week")["quantity_sold"].mean().reset_index()
dow_df.columns = ["Day", "Avg Quantity"]
dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
dow_df["Day Label"] = [dow_labels[int(d)] for d in dow_df["Day"]]

fig_dow = px.bar(
    dow_df,
    x="Day Label",
    y="Avg Quantity",
    text="Avg Quantity",
    color="Avg Quantity",
    color_continuous_scale=["#1a1a6c", "#667eea", "#f093fb"],
)
fig_dow.update_traces(
    texttemplate="%{text:.0f}",
    textposition="outside",
    marker_line_width=0,
)
fig_dow.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=300,
    showlegend=False,
    coloraxis_showscale=False,
    margin=dict(l=40, r=20, t=20, b=40),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", title=""),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Avg Quantity"),
)
st.plotly_chart(fig_dow, use_container_width=True)

# ─── Footer ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#555; padding:16px;'>"
    "💊 <b>PharmaCast</b> — LSTM-based Predictive Stock Analytics for Pharmacy Retail<br>"
    "<small>Built with Streamlit, TensorFlow, and Plotly | v1.0</small>"
    "</div>",
    unsafe_allow_html=True,
)
