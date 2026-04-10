"""
Page 2: Exploratory Data Analysis (EDA)
8 charts + SQL Explorer with 10 preset queries.
"""
import os, sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import PRODUCTS, SEASON_COLORS
from src.data_preprocessing import load_and_preprocess, prepare_product_data
from src.db import init_db, db_status, execute_custom, PRESET_QUERIES

st.set_page_config(page_title="PharmaCast — EDA", page_icon="📊", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:linear-gradient(135deg,#0f0c29 0%,#1a1a2e 50%,#16213e 100%);}
.section-header{font-size:1.25rem;font-weight:700;color:#e2e8f0;padding:8px 0 14px;border-bottom:2px solid rgba(102,126,234,.35);margin-bottom:18px;}
.sql-result-header{background:rgba(102,126,234,.12);border-radius:8px;padding:10px 14px;color:#a5b4fc;font-size:.85rem;margin-bottom:8px;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#1a1a2e 0%,#0f0c29 100%);}
#MainMenu{visibility:hidden;}footer{visibility:hidden;}header{visibility:hidden;}
</style>""", unsafe_allow_html=True)

DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=40, r=20, t=40, b=40),
    font=dict(family="Inter"),
)

st.markdown(
    "<h1 style='background:linear-gradient(135deg,#667eea,#764ba2);-webkit-background-clip:text;"
    "-webkit-text-fill-color:transparent;font-size:2.2rem;font-weight:800;margin-bottom:0;'>"
    "📊 Exploratory Data Analysis</h1>"
    "<p style='color:#8b8fa3;margin-top:0;'>Understand demand patterns before building the forecasting model.</p>",
    unsafe_allow_html=True,
)

# ─── Load ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    from config import DATA_FILE
    return load_and_preprocess(DATA_FILE)

df = load_data()
df["date"] = pd.to_datetime(df["date"])
df["year"]  = df["date"].dt.year
df["month_str"] = df["date"].dt.to_period("M").astype(str)
# Ensure revenue column exists (it's in the generated CSV, but guard against edge cases)
if "revenue" not in df.columns:
    df["revenue"] = df["quantity_sold"] * df.get("unit_price", pd.Series(0, index=df.index))

# ─── 1. Monthly Sales Trend ───────────────────────────────────────────
st.markdown('<div class="section-header">📅 1. Monthly Sales Trend (All Products)</div>', unsafe_allow_html=True)

monthly = df.groupby("month_str").agg(
    total_units   = ("quantity_sold", "sum"),
    total_revenue = ("revenue", "sum"),
).reset_index()
monthly.columns = ["Month", "Total Units", "Total Revenue"]

fig_monthly = make_subplots(specs=[[{"secondary_y": True}]])
fig_monthly.add_trace(go.Bar(
    x=monthly["Month"], y=monthly["Total Units"],
    name="Units Sold", marker_color="rgba(102,126,234,0.7)",
), secondary_y=False)
fig_monthly.add_trace(go.Scatter(
    x=monthly["Month"], y=monthly["Total Revenue"],
    name="Revenue (₹)", line=dict(color="#f093fb", width=2.5),
    mode="lines+markers", marker=dict(size=4),
), secondary_y=True)
fig_monthly.update_layout(height=380, **DARK_LAYOUT,
    legend=dict(orientation="h", y=1.05, x=1, xanchor="right"))
fig_monthly.update_xaxes(gridcolor="rgba(255,255,255,.05)")
fig_monthly.update_yaxes(gridcolor="rgba(255,255,255,.05)", title_text="Units", secondary_y=False)
fig_monthly.update_yaxes(title_text="Revenue (₹)", secondary_y=True)
st.plotly_chart(fig_monthly, use_container_width=True)

# ─── 2. Top 10 Products ───────────────────────────────────────────────
st.markdown('<div class="section-header">🏆 2. Top Products by Total Sales</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    top_products = (df.groupby("product_name")["quantity_sold"]
                    .sum().reset_index()
                    .sort_values("quantity_sold", ascending=True).tail(12))
    fig_top = go.Figure(go.Bar(
        x=top_products["quantity_sold"], y=top_products["product_name"],
        orientation="h",
        marker=dict(
            color=top_products["quantity_sold"],
            colorscale=[[0, "#1a1a6c"], [0.5, "#667eea"], [1, "#f093fb"]],
        ),
        text=top_products["quantity_sold"].apply(lambda x: f"{x:,}"),
        textposition="outside",
    ))
    fig_top.update_layout(height=420, xaxis_title="Total Units Sold",
                          yaxis_title="", **DARK_LAYOUT)
    fig_top.update_xaxes(gridcolor="rgba(255,255,255,.05)")
    st.plotly_chart(fig_top, use_container_width=True)

with c2:
    # Revenue by category
    cat_rev = (df.groupby("category")["revenue"].sum().reset_index()
               .sort_values("revenue", ascending=False))
    cat_rev.columns = ["Category", "Revenue"]
    fig_cat = px.pie(
        cat_rev, values="Revenue", names="Category",
        color_discrete_sequence=px.colors.sequential.Plasma,
        hole=0.45,
    )
    fig_cat.update_layout(height=420, title="Revenue by Category", **DARK_LAYOUT)
    fig_cat.update_traces(textinfo="percent+label", textfont_size=12)
    st.plotly_chart(fig_cat, use_container_width=True)

# ─── 3. Season-wise Demand by Category ───────────────────────────────
st.markdown('<div class="section-header">🌦️ 3. Season-wise Demand by Category</div>', unsafe_allow_html=True)

season_cat = (df.groupby(["season", "category"])["quantity_sold"]
              .mean().reset_index())
season_cat.columns = ["Season", "Category", "Avg Daily Demand"]

fig_sc = px.bar(
    season_cat, x="Season", y="Avg Daily Demand",
    color="Category", barmode="group",
    color_discrete_sequence=px.colors.qualitative.Vivid,
)
fig_sc.update_layout(height=380, **DARK_LAYOUT,
    legend=dict(orientation="h", y=1.05, x=1, xanchor="right"))
fig_sc.update_xaxes(gridcolor="rgba(255,255,255,.05)")
fig_sc.update_yaxes(gridcolor="rgba(255,255,255,.05)", title_text="Avg Daily Demand")
st.plotly_chart(fig_sc, use_container_width=True)

# ─── 4. Daily / Weekly Line Chart (selected product) ─────────────────
st.markdown('<div class="section-header">📈 4. Daily & Weekly Demand Trend (Single Product)</div>', unsafe_allow_html=True)

product_opts = {f"{v['name']} ({k})": k for k, v in PRODUCTS.items()}
sel_label    = st.selectbox("Select Product", list(product_opts.keys()))
sel_pid      = product_opts[sel_label]

pdata = prepare_product_data(df, sel_pid)
pdata["MA_7"]  = pdata["quantity_sold"].rolling(7).mean()
pdata["MA_30"] = pdata["quantity_sold"].rolling(30).mean()

fig_daily = go.Figure()
fig_daily.add_trace(go.Scatter(
    x=pdata["date"], y=pdata["quantity_sold"],
    name="Daily Sales", mode="lines",
    line=dict(color="rgba(102,126,234,0.25)", width=1),
))
fig_daily.add_trace(go.Scatter(
    x=pdata["date"], y=pdata["MA_7"],
    name="7-day MA", mode="lines",
    line=dict(color="#667eea", width=2.5),
))
fig_daily.add_trace(go.Scatter(
    x=pdata["date"], y=pdata["MA_30"],
    name="30-day MA", mode="lines",
    line=dict(color="#f093fb", width=2.5),
))
fig_daily.update_layout(height=380, hovermode="x unified", **DARK_LAYOUT,
    legend=dict(orientation="h", y=1.05, x=1, xanchor="right"),
    yaxis_title="Quantity Sold")
fig_daily.update_xaxes(gridcolor="rgba(255,255,255,.05)")
fig_daily.update_yaxes(gridcolor="rgba(255,255,255,.05)")
st.plotly_chart(fig_daily, use_container_width=True)

# ─── 5. Fast vs Slow Movers ───────────────────────────────────────────
st.markdown('<div class="section-header">⚡ 5. Fast-Moving vs Slow-Moving Products</div>', unsafe_allow_html=True)

avg_daily_by_product = (df.groupby(["product_name", "category"])["quantity_sold"]
                        .mean().reset_index())
avg_daily_by_product.columns = ["Product", "Category", "Avg Daily Demand"]
avg_daily_by_product = avg_daily_by_product.sort_values("Avg Daily Demand", ascending=False)

fig_movers = px.bar(
    avg_daily_by_product, x="Product", y="Avg Daily Demand",
    color="Avg Daily Demand",
    color_continuous_scale=["#1a1a6c", "#667eea", "#f093fb"],
    text="Avg Daily Demand", hover_data=["Category"],
)
fig_movers.update_traces(texttemplate="%{text:.0f}", textposition="outside")
fig_movers.update_layout(height=380, coloraxis_showscale=False,
                         xaxis_tickangle=-30, **DARK_LAYOUT)
fig_movers.update_xaxes(gridcolor="rgba(255,255,255,.05)")
fig_movers.update_yaxes(gridcolor="rgba(255,255,255,.05)")
st.plotly_chart(fig_movers, use_container_width=True)

# ─── 6. Demand Distribution ───────────────────────────────────────────
st.markdown('<div class="section-header">📊 6. Demand Distribution & Heatmap</div>', unsafe_allow_html=True)

d1, d2 = st.columns(2)
with d1:
    fig_hist = px.histogram(
        df, x="quantity_sold", nbins=60,
        color_discrete_sequence=["#667eea"],
        title="Demand Distribution",
    )
    fig_hist.update_layout(height=360, **DARK_LAYOUT, showlegend=False)
    fig_hist.update_xaxes(gridcolor="rgba(255,255,255,.05)", title="Qty Sold")
    fig_hist.update_yaxes(gridcolor="rgba(255,255,255,.05)", title="Count")
    st.plotly_chart(fig_hist, use_container_width=True)

with d2:
    # Monthly heatmap for selected product
    pdata_copy = pdata.copy()
    pdata_copy["year"]  = pdata_copy["date"].dt.year
    pdata_copy["month"] = pdata_copy["date"].dt.month
    pivot = (pdata_copy.pivot_table(values="quantity_sold", index="year",
                                     columns="month", aggfunc="mean").round(0))
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    # Fill NaN with 0 before casting to int to avoid ValueError on sparse pivots
    pivot_vals = np.nan_to_num(pivot.values, nan=0).astype(int)
    fig_heat = go.Figure(go.Heatmap(
        z=pivot_vals,
        x=[month_labels[m-1] for m in pivot.columns],
        y=[str(y) for y in pivot.index],
        colorscale=[[0,"#0f0c29"],[0.5,"#667eea"],[1,"#f093fb"]],
        text=pivot_vals,
        texttemplate="%{text}",
        textfont=dict(size=11, color="white"),
        hovertemplate="Year:%{y} Month:%{x} Avg:%{z:.0f}<extra></extra>",
    ))
    fig_heat.update_layout(height=360, title=f"Monthly Heatmap — {PRODUCTS[sel_pid]['name']}", **DARK_LAYOUT)
    st.plotly_chart(fig_heat, use_container_width=True)

# ─── 7. Day-of-Week Pattern ───────────────────────────────────────────
st.markdown('<div class="section-header">📅 7. Day-of-Week Demand Pattern</div>', unsafe_allow_html=True)

dow_df = pdata.groupby("day_of_week")["quantity_sold"].mean().reset_index()
dow_labels = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
dow_df["Day"] = [dow_labels[int(d)] for d in dow_df["day_of_week"]]

fig_dow = px.bar(
    dow_df, x="Day", y="quantity_sold",
    text="quantity_sold",
    color="quantity_sold",
    color_continuous_scale=["#1a1a6c","#667eea","#f093fb"],
)
fig_dow.update_traces(texttemplate="%{text:.0f}", textposition="outside")
fig_dow.update_layout(height=340, coloraxis_showscale=False, **DARK_LAYOUT,
                      yaxis_title="Avg Qty Sold")
fig_dow.update_xaxes(gridcolor="rgba(255,255,255,.05)")
fig_dow.update_yaxes(gridcolor="rgba(255,255,255,.05)")
st.plotly_chart(fig_dow, use_container_width=True)

# ─── 8. Year-over-Year Comparison ────────────────────────────────────
st.markdown('<div class="section-header">📆 8. Year-over-Year Sales Comparison</div>', unsafe_allow_html=True)

yoy = df.groupby(["year", "product_name"])["quantity_sold"].sum().reset_index()
yoy.columns = ["Year", "Product", "Total Units"]
fig_yoy = px.bar(
    yoy, x="Year", y="Total Units", color="Product",
    barmode="group",
    color_discrete_sequence=px.colors.qualitative.Vivid,
)
fig_yoy.update_layout(height=400, **DARK_LAYOUT,
    legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center", font=dict(size=10)))
fig_yoy.update_xaxes(gridcolor="rgba(255,255,255,.05)", type="category")
fig_yoy.update_yaxes(gridcolor="rgba(255,255,255,.05)")
st.plotly_chart(fig_yoy, use_container_width=True)

# ─── SQL Explorer ─────────────────────────────────────────────────────
st.markdown('<div class="section-header">🗄️ SQL Explorer</div>', unsafe_allow_html=True)
st.markdown(
    "<div style='background:rgba(102,126,234,0.08);border-radius:10px;padding:14px;margin-bottom:16px;"
    "border:1px solid rgba(102,126,234,0.2);color:#94a3b8;font-size:.85rem;'>"
    "💡 Run SQL queries directly against the pharmacy SQLite database. "
    "Choose a preset query or write your own SQL below."
    "</div>",
    unsafe_allow_html=True,
)

# Initialise DB if needed
try:
    status = db_status()
    if status.get("status") != "ok":
        with st.spinner("Initialising database..."):
            init_db()
        status = db_status()
    st.markdown(
        f"<div class='sql-result-header'>🗄️ Database: {status['rows']:,} rows | "
        f"{status['products']} products | {status['date_from']} → {status['date_to']}</div>",
        unsafe_allow_html=True,
    )
except Exception as e:
    st.warning(f"Database not initialised. Run `python src/db.py` first.\n\n{e}")

# Preset selector
selected_preset = st.selectbox("📋 Preset Query", ["(Custom SQL)"] + list(PRESET_QUERIES.keys()))

if selected_preset == "(Custom SQL)":
    default_sql = "SELECT product_name, SUM(quantity_sold) AS total_sold\nFROM sales\nGROUP BY product_name\nORDER BY total_sold DESC;"
else:
    default_sql = PRESET_QUERIES[selected_preset]

sql_input = st.text_area("✏️ SQL Query", value=default_sql, height=160,
                          help="Write any SELECT statement against the `sales` table.")

if st.button("▶️ Run Query", type="primary"):
    try:
        with st.spinner("Executing..."):
            result_df = execute_custom(sql_input)
        st.markdown(
            f"<div class='sql-result-header'>✅ Query returned {len(result_df):,} rows</div>",
            unsafe_allow_html=True,
        )
        st.dataframe(result_df, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"❌ SQL Error: {e}")

st.markdown("---")
st.markdown("<div style='text-align:center;color:#555;font-size:.75rem;padding:10px;'>"
            "💊 PharmaCast v2.0 | EDA & SQL Explorer</div>", unsafe_allow_html=True)
