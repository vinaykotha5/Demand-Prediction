"""
Page 5: Reports — Export CSV / Excel
"""
import os, sys
import json
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from io import BytesIO

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import PRODUCTS, FORECAST_HORIZONS
from src.data_preprocessing import load_and_preprocess, prepare_product_data
from src.stock_recommender import get_season, recommend_stock, generate_all_alerts
from src.inventory import get_all_stock_levels

st.set_page_config(page_title="PharmaCast — Reports", page_icon="📥", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:linear-gradient(135deg,#0f0c29 0%,#1a1a2e 50%,#16213e 100%);}
.kpi-card{background:linear-gradient(135deg,rgba(30,33,48,.92),rgba(44,47,68,.92));border:1px solid rgba(255,255,255,.08);border-radius:16px;padding:20px 24px;text-align:center;backdrop-filter:blur(10px);box-shadow:0 8px 32px rgba(0,0,0,.3);}
.kpi-value{font-size:2rem;font-weight:800;background:linear-gradient(135deg,#667eea,#764ba2);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:4px 0;}
.kpi-label{font-size:.8rem;color:#8b8fa3;text-transform:uppercase;letter-spacing:1.2px;font-weight:500;}
.kpi-icon{font-size:1.8rem;margin-bottom:4px;}
.section-header{font-size:1.25rem;font-weight:700;color:#e2e8f0;padding:8px 0 14px;border-bottom:2px solid rgba(102,126,234,.35);margin-bottom:18px;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#1a1a2e 0%,#0f0c29 100%);}
#MainMenu{visibility:hidden;}footer{visibility:hidden;}header{visibility:hidden;}
</style>""", unsafe_allow_html=True)

st.markdown(
    "<h1 style='background:linear-gradient(135deg,#667eea,#764ba2);-webkit-background-clip:text;"
    "-webkit-text-fill-color:transparent;font-size:2.2rem;font-weight:800;margin-bottom:0;'>"
    "📥 Reports & Exports</h1>"
    "<p style='color:#8b8fa3;margin-top:0;'>Download forecast data, stock recommendations, and alerts as CSV or Excel.</p>",
    unsafe_allow_html=True,
)

# ─── Data loading ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    from config import DATA_FILE
    return load_and_preprocess(DATA_FILE)

df = load_data()

current_month  = datetime.now().month
current_season = get_season(current_month)
stock_levels   = get_all_stock_levels()

# ─── Controls ────────────────────────────────────────────────────────
st.markdown('<div class="section-header">⚙️ Report Settings</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    horizon = st.selectbox("📅 Forecast Horizon", FORECAST_HORIZONS,
                           format_func=lambda x: f"{x} days", index=1)
with c2:
    report_date = st.date_input("📆 Report Date", value=datetime.today())

st.markdown("")

# ─── Build data frames ────────────────────────────────────────────────
@st.cache_data
def build_forecast_df(horizon, season):
    rows = []
    for pid, pinfo in PRODUCTS.items():
        pdata  = prepare_product_data(df, pid)
        qty    = pdata["quantity_sold"].values
        last30 = qty[-30:]
        mean_v = np.mean(last30)
        trend  = np.polyfit(range(len(last30)), last30, 1)[0]
        preds  = [max(1, int(mean_v + trend * d)) for d in range(horizon)]
        total  = int(np.sum(preds))
        avg_d  = float(np.mean(qty[-30:]))
        lead   = pinfo.get("lead_time_days", 5)
        rec    = recommend_stock(total, season, pinfo["category"],
                                  avg_daily_demand=avg_d, lead_time_days=lead)
        rows.append({
            "Product ID":         pid,
            "Product Name":       pinfo["name"],
            "Category":           pinfo["category"].title(),
            "Unit Price (₹)":     pinfo["unit_price"],
            f"Forecast ({horizon}d)": total,
            "Avg Daily Demand":   round(avg_d, 1),
            "Safety Stock":       rec["safety_stock"],
            "Reorder Point":      rec["reorder_point"],
            "Recommended Stock":  rec["recommended_stock"],
            "Buffer %":           rec["buffer_pct"],
            "Seasonal Factor":    rec["seasonal_factor"],
            "Lead Time (days)":   lead,
            "Current Stock":      stock_levels.get(pid, 0),
        })
    return pd.DataFrame(rows)

@st.cache_data
def build_alerts_df(horizon, season):
    from src.stock_recommender import generate_all_alerts
    recs = {}
    for pid, pinfo in PRODUCTS.items():
        pdata  = prepare_product_data(df, pid)
        qty    = pdata["quantity_sold"].values
        last30 = qty[-30:]
        total  = int(np.sum([max(1, int(np.mean(last30) + np.polyfit(range(30), last30, 1)[0] * d)) for d in range(horizon)]))
        avg_d  = float(np.mean(qty[-30:]))
        recs[pid] = recommend_stock(total, season, pinfo["category"],
                                     avg_daily_demand=avg_d,
                                     lead_time_days=pinfo.get("lead_time_days", 5))
    alerts = generate_all_alerts(stock_levels, recs)
    if not alerts:
        return pd.DataFrame([{"Status": "All products healthy", "Message": "No alerts generated."}])
    return pd.DataFrame([{
        "Product":           a["product"],
        "Alert Type":        a["type"],
        "Severity":          a["severity"],
        "Current Stock":     a["current_stock"],
        "Recommended Stock": a["recommended_stock"],
        "Reorder Point":     a.get("reorder_point", "N/A"),
        "Deficit/Excess":    a.get("deficit", a.get("excess", "N/A")),
        "Message":           a["message"],
    } for a in alerts])

@st.cache_data
def build_daily_sales_df():
    rows = []
    for pid, pinfo in PRODUCTS.items():
        pdata = prepare_product_data(df, pid)
        for _, row in pdata.iterrows():
            rows.append({
                "Date":         str(row["date"])[:10],
                "Product ID":   pid,
                "Product Name": pinfo["name"],
                "Category":     pinfo["category"].title(),
                "Season":       row.get("season", ""),
                "Qty Sold":     int(row["quantity_sold"]),
                "Revenue (₹)":  round(row.get("revenue", row["quantity_sold"] * pinfo["unit_price"]), 2),
            })
    return pd.DataFrame(rows)

forecast_df     = build_forecast_df(horizon, current_season)
alerts_df       = build_alerts_df(horizon, current_season)
daily_sales_df  = build_daily_sales_df()

# ─── Download helpers ─────────────────────────────────────────────────
def to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def to_excel(sheets: dict) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for sheet_name, sheet_df in sheets.items():
            sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
    return buf.getvalue()

ts = datetime.now().strftime("%Y%m%d_%H%M")

# ─── Export Cards ────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Available Exports</div>', unsafe_allow_html=True)

e1, e2, e3, e4 = st.columns(4)

with e1:
    st.markdown("""<div class="kpi-card">
      <div class="kpi-icon">📈</div>
      <div class="kpi-value" style="font-size:1.2rem;">Forecasts</div>
      <div class="kpi-label">Per-product demand + stock recs</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("")
    st.download_button(
        "📥 Download CSV",
        data=to_csv(forecast_df),
        file_name=f"pharmacast_forecasts_{horizon}d_{ts}.csv",
        mime="text/csv",
        use_container_width=True,
    )

with e2:
    st.markdown("""<div class="kpi-card">
      <div class="kpi-icon">⚠️</div>
      <div class="kpi-value" style="font-size:1.2rem;">Alerts</div>
      <div class="kpi-label">Current stock alert report</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("")
    st.download_button(
        "📥 Download CSV",
        data=to_csv(alerts_df),
        file_name=f"pharmacast_alerts_{ts}.csv",
        mime="text/csv",
        use_container_width=True,
    )

with e3:
    st.markdown("""<div class="kpi-card">
      <div class="kpi-icon">📋</div>
      <div class="kpi-value" style="font-size:1.2rem;">Sales Data</div>
      <div class="kpi-label">Full historical daily sales</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("")
    st.download_button(
        "📥 Download CSV",
        data=to_csv(daily_sales_df),
        file_name=f"pharmacast_sales_{ts}.csv",
        mime="text/csv",
        use_container_width=True,
    )

with e4:
    st.markdown("""<div class="kpi-card">
      <div class="kpi-icon">📊</div>
      <div class="kpi-value" style="font-size:1.2rem;">Full Report</div>
      <div class="kpi-label">All sheets in one Excel file</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("")
    excel_bytes = to_excel({
        "Forecasts":   forecast_df,
        "Alerts":      alerts_df,
        "Daily Sales": daily_sales_df,
    })
    st.download_button(
        "📥 Download Excel",
        data=excel_bytes,
        file_name=f"pharmacast_full_report_{ts}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

# ─── Data Previews ────────────────────────────────────────────────────
st.markdown("")
st.markdown('<div class="section-header">🔍 Data Previews</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📈 Forecast + Stock Recommendations", "⚠️ Alerts", "📋 Daily Sales (sample)"])

with tab1:
    st.dataframe(forecast_df, use_container_width=True, hide_index=True)

with tab2:
    st.dataframe(alerts_df, use_container_width=True, hide_index=True)

with tab3:
    st.caption("Showing last 200 rows of historical sales data")
    st.dataframe(daily_sales_df.tail(200), use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("<div style='text-align:center;color:#555;font-size:.75rem;padding:10px;'>"
            "💊 PharmaCast v2.0 | Reports Page</div>", unsafe_allow_html=True)
