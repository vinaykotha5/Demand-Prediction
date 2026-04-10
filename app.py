"""
PharmaCast — Landing Page
Streamlit multi-page app entry point.

Run: streamlit run app.py
"""
import os
import sys
import streamlit as st
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import STORE_NAME, PRODUCTS, STORE_ID
from src.stock_recommender import get_season, get_season_emoji

# ─── Page Config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="PharmaCast — Pharmacy Intelligence",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Shared CSS (injected into every page via app.py) ─────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .stApp {
      background: linear-gradient(160deg, #0d1117 0%, #0a1628 40%, #071a20 100%);
  }

  /* ── KPI Cards ── */
  .kpi-card {
      background: linear-gradient(135deg, rgba(13,25,40,0.95), rgba(10,30,45,0.95));
      border: 1px solid rgba(20,184,166,0.25);
      border-radius: 16px;
      padding: 20px 24px;
      text-align: center;
      backdrop-filter: blur(12px);
      box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(20,184,166,0.1);
      transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s;
  }
  .kpi-card:hover {
      transform: translateY(-4px);
      box-shadow: 0 16px 48px rgba(20,184,166,0.2);
      border-color: rgba(20,184,166,0.5);
  }
  .kpi-value {
      font-size: 2.2rem;
      font-weight: 800;
      background: linear-gradient(135deg, #14b8a6, #06b6d4, #10b981);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin: 4px 0;
  }
  .kpi-label {
      font-size: 0.78rem;
      color: #64748b;
      text-transform: uppercase;
      letter-spacing: 1.5px;
      font-weight: 600;
  }
  .kpi-icon   { font-size: 1.8rem; margin-bottom: 6px; }
  .kpi-delta  { font-size: 0.78rem; margin-top: 4px; }
  .kpi-delta.positive { color: #10b981; }
  .kpi-delta.negative { color: #f87171; }

  /* ── Section Headers ── */
  .section-header {
      font-size: 1.15rem;
      font-weight: 700;
      color: #e2e8f0;
      padding: 8px 0 14px;
      border-bottom: 2px solid rgba(20,184,166,0.3);
      margin-bottom: 18px;
      letter-spacing: 0.3px;
  }

  /* ── Alert cards ── */
  .alert-critical {
      background: linear-gradient(135deg, rgba(220,38,38,0.15), rgba(220,38,38,0.05));
      border-left: 4px solid #ef4444;
      border-radius: 10px; padding: 12px 16px; margin: 8px 0; color: #fca5a5;
  }
  .alert-warning {
      background: linear-gradient(135deg, rgba(245,158,11,0.15), rgba(245,158,11,0.05));
      border-left: 4px solid #f59e0b;
      border-radius: 10px; padding: 12px 16px; margin: 8px 0; color: #fcd34d;
  }
  .alert-info {
      background: linear-gradient(135deg, rgba(20,184,166,0.15), rgba(20,184,166,0.05));
      border-left: 4px solid #14b8a6;
      border-radius: 10px; padding: 12px 16px; margin: 8px 0; color: #5eead4;
  }

  /* ── Stat rows ── */
  .stat-row {
      display: flex; justify-content: space-between;
      padding: 8px 14px; margin: 4px 0; border-radius: 8px;
      background: rgba(20,184,166,0.04);
      border: 1px solid rgba(20,184,166,0.08);
  }
  .stat-label { color: #64748b; font-size: 0.88rem; }
  .stat-value { color: #e2e8f0; font-weight: 600; font-size: 0.88rem; }

  /* ── SQL Explorer ── */
  .sql-result-header {
      background: rgba(20,184,166,0.1);
      border-radius: 8px; padding: 10px 14px;
      color: #5eead4; font-size: 0.85rem; margin-bottom: 8px;
  }

  /* ── Hero ── */
  .hero-title {
      font-size: 3.2rem; font-weight: 800;
      background: linear-gradient(135deg, #14b8a6 0%, #06b6d4 50%, #10b981 100%);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      line-height: 1.15; margin-bottom: 0.4rem;
  }
  .hero-sub {
      font-size: 1.05rem; color: #64748b; margin-bottom: 2rem;
  }
  .feature-card {
      background: linear-gradient(135deg, rgba(13,25,40,0.9), rgba(10,30,45,0.9));
      border: 1px solid rgba(20,184,166,0.2);
      border-radius: 16px; padding: 24px; text-align: center;
      backdrop-filter: blur(10px);
      transition: transform 0.25s, border-color 0.25s, box-shadow 0.25s;
  }
  .feature-card:hover {
      transform: translateY(-6px);
      border-color: rgba(20,184,166,0.55);
      box-shadow: 0 20px 50px rgba(20,184,166,0.15);
  }
  .feature-icon  { font-size: 2.2rem; margin-bottom: 12px; }
  .feature-title { font-size: 0.95rem; font-weight: 700; color: #e2e8f0; margin-bottom: 8px; }
  .feature-desc  { font-size: 0.8rem; color: #64748b; line-height: 1.6; }

  /* Sidebar */
  [data-testid="stSidebar"] {
      background: linear-gradient(180deg, #0a1628 0%, #0d1117 100%);
      border-right: 1px solid rgba(20,184,166,0.15);
  }
  #MainMenu { visibility: hidden; }
  footer     { visibility: hidden; }
  header     { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💊 PharmaCast")
    st.markdown(f"*{STORE_NAME}* · `{STORE_ID}`")
    st.markdown("---")

    current_month  = datetime.now().month
    current_season = get_season(current_month)
    season_emoji   = get_season_emoji(current_season)

    st.markdown(f"### {season_emoji} {current_season}")
    st.markdown(f"📅 {datetime.now().strftime('%B %d, %Y')}")
    st.markdown("---")
    st.markdown(
        "<div style='color:#555; font-size:0.75rem; text-align:center;'>"
        "💊 PharmaCast v2.0<br>"
        "LSTM · XGBoost · SQLite<br>"
        "© 2025 PharmaCast"
        "</div>",
        unsafe_allow_html=True,
    )


# ─── Landing Page Content ─────────────────────────────────────────────
st.markdown(
    "<div class='hero-title'>💊 PharmaCast</div>"
    "<div class='hero-sub'>AI-powered pharmacy demand forecasting & inventory intelligence platform</div>",
    unsafe_allow_html=True,
)

# Quick stats bar
n_products = len(PRODUCTS)
q1, q2, q3, q4 = st.columns(4)
with q1:
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-icon">💊</div>
      <div class="kpi-value">{n_products}</div>
      <div class="kpi-label">Products Tracked</div>
    </div>""", unsafe_allow_html=True)
with q2:
    st.markdown("""
    <div class="kpi-card">
      <div class="kpi-icon">📅</div>
      <div class="kpi-value">3 yr</div>
      <div class="kpi-label">Historical Data</div>
    </div>""", unsafe_allow_html=True)
with q3:
    st.markdown("""
    <div class="kpi-card">
      <div class="kpi-icon">🧠</div>
      <div class="kpi-value">LSTM</div>
      <div class="kpi-label">Forecast Engine</div>
    </div>""", unsafe_allow_html=True)
with q4:
    st.markdown("""
    <div class="kpi-card">
      <div class="kpi-icon">🗄️</div>
      <div class="kpi-value">SQLite</div>
      <div class="kpi-label">Data Layer</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Feature cards
st.markdown('<div class="section-header">🚀 Navigate the Dashboard</div>', unsafe_allow_html=True)

features = [
    ("🏠", "Overview",       "Store-wide KPIs, all-product stock table, and live alert panel."),
    ("📊", "EDA",            "Full exploratory data analysis with 8 interactive charts and SQL Explorer."),
    ("📈", "Forecasting",    "Per-product LSTM demand forecasts with Naive vs XGBoost vs LSTM comparison."),
    ("📦", "Inventory",      "Edit current stock levels, view reorder points, and safety stock formulas."),
    ("📥", "Reports",        "Export forecasts, stock recommendations, and alerts to CSV or Excel."),
]

cols = st.columns(5)
for col, (icon, title, desc) in zip(cols, features):
    with col:
        st.markdown(f"""
        <div class="feature-card">
          <div class="feature-icon">{icon}</div>
          <div class="feature-title">{title}</div>
          <div class="feature-desc">{desc}</div>
        </div>""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#555; padding:16px;'>"
    "💊 <b>PharmaCast</b> — LSTM-based Predictive Stock Analytics for Pharmacy Retail<br>"
    "<small>Built with Streamlit · TensorFlow · XGBoost · SQLite · Plotly | v2.0</small>"
    "</div>",
    unsafe_allow_html=True,
)
