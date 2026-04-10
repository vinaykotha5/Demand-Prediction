# 💊 PharmaCast — Pharmacy Demand Prediction & Stock Analytics

> **AI-powered demand forecasting and inventory intelligence for pharmacy retail stores.**
> Built with LSTM neural networks, XGBoost, SQLite, and Streamlit.

---

## 📋 Table of Contents
1. [Project Objective](#-project-objective)
2. [Problem Statement](#-problem-statement)
3. [Dataset Description](#-dataset-description)
4. [Features Used](#-features-used)
5. [Project Workflow](#-project-workflow)
6. [SQL Queries](#-sql-queries)
7. [Model Comparison](#-model-comparison)
8. [Evaluation Metrics](#-evaluation-metrics)
9. [Dashboard Pages](#-dashboard-pages)
10. [Business Impact](#-business-impact)
11. [How to Run](#-how-to-run)
12. [Future Scope](#-future-scope)

---

## 🎯 Project Objective

Build a predictive analytics system for pharmacy retail that:
- **Forecasts demand** for each medicine over 7, 30, and 60-day horizons
- **Recommends optimal stock levels** using safety stock and reorder point formulas
- **Alerts pharmacy managers** when stock levels are critically low, overstocked, or at seasonal risk
- **Provides EDA** and business insights on historical sales patterns

---

## ❓ Problem Statement

Small and mid-size pharmacy stores often face two costly problems:
1. **Stock-outs** — key medicines running out of stock, causing lost sales and patient risk
2. **Overstocking** — excess inventory leading to expired medicines and capital lock-up

There is no scalable, data-driven system to proactively predict demand and recommend the right stock level per product, per season.

**PharmaCast solves this** by combining time-series forecasting (LSTM), feature engineering (lags, rolling stats, seasonality), and a business-rules recommendation engine into a single interactive dashboard.

---

## 📊 Dataset Description

**Location:** `data/pharmacy_sales.csv`  
**Period:** January 2022 – December 2024 (3 years)  
**Records:** ~13,200 rows (12 products × ~1,096 days)

### Columns

| Column              | Type    | Description                                     |
|---------------------|---------|-------------------------------------------------|
| `date`              | Date    | Sales date (daily granularity)                  |
| `store_id`          | String  | Pharmacy store identifier                       |
| `product_id`        | String  | Medicine code (P001–P012)                       |
| `product_name`      | String  | Full product name                               |
| `category`          | String  | Therapeutic category (fever, hydration, etc.)   |
| `season`            | String  | Indian season (Summer/Monsoon/Post-Monsoon/Winter) |
| `quantity_sold`     | Integer | Units sold that day                             |
| `unit_price`        | Float   | Price per unit (₹)                              |
| `revenue`           | Float   | Daily revenue (quantity × unit_price)           |
| `current_stock`     | Integer | End-of-day stock level (simulated)              |
| `lead_time_days`    | Integer | Supplier lead time in days (3–7)                |
| `expiry_risk`       | Binary  | 1 if stock > 60 days of average demand          |
| `is_festival_month` | Binary  | 1 if month has a major Indian festival          |

### Products (12 medicines across 8 categories)
| ID   | Name                | Category    | Unit Price |
|------|---------------------|-------------|------------|
| P001 | Paracetamol 500mg   | fever       | ₹15        |
| P002 | ORS Powder          | hydration   | ₹12        |
| P003 | Cough Syrup 100ml   | respiratory | ₹85        |
| P004 | Cetirizine 10mg     | allergy     | ₹25        |
| P005 | Dolo 650            | fever       | ₹30        |
| P006 | Azithromycin 500mg  | antibiotic  | ₹120       |
| P007 | Vitamin C 500mg     | supplement  | ₹45        |
| P008 | Ibuprofen 400mg     | painkiller  | ₹20        |
| P009 | Amoxicillin 250mg   | antibiotic  | ₹65        |
| P010 | Electral Powder     | hydration   | ₹18        |
| P011 | Insulin Glargine    | diabetes    | ₹450       |
| P012 | Antihistamine Tabs  | allergy     | ₹35        |

---

## 🔧 Features Used

### Time-Based
| Feature       | Description                         |
|---------------|-------------------------------------|
| `month`       | Month number (1–12)                 |
| `day_of_week` | Weekday (0=Mon, 6=Sun)              |
| `day_of_month`| Day of month (1–31)                 |
| `quarter`     | Quarter (1–4)                       |
| `is_weekend`  | 1 if Saturday or Sunday             |
| `week_sin`    | Cyclical sin encoding of week       |
| `week_cos`    | Cyclical cos encoding of week       |

### Seasonal / Business
| Feature                  | Description                              |
|--------------------------|------------------------------------------|
| `season_encoded`         | Indian season as integer (0–3)           |
| `is_festival_month`      | 1 if major Indian festival in that month |
| `is_peak_disease_season` | 1 if category peaks in this season       |

### Lag Features (Time-Series)
| Feature | Description                       |
|---------|-----------------------------------|
| `lag_1` | Sales from 1 day ago              |
| `lag_7` | Sales from 7 days ago (last week) |
| `lag_30`| Sales from 30 days ago            |

### Rolling Statistics
| Feature          | Description                    |
|------------------|--------------------------------|
| `rolling_mean_7` | 7-day rolling average          |
| `rolling_mean_30`| 30-day rolling average         |
| `rolling_std_7`  | 7-day rolling standard deviation|

**Total features: 17** (used by LSTM); XGBoost uses the same features in tabular format.

---

## 🔄 Project Workflow

```
Raw CSV Data
    │
    ▼
generate_dataset.py  ──► data/pharmacy_sales.csv  (enriched dataset)
    │
    ▼
src/db.py            ──► data/pharmacy.db          (SQLite layer)
    │
    ▼
src/data_preprocessing.py
    │  Load → Time features → Lag features → Rolling stats → Scale → Sequences
    ▼
src/model.py         ──► LSTM (or Bidirectional LSTM)
src/evaluator.py     ──► Naive baseline, XGBoost baseline, LSTM evaluation
    │
    ▼
src/train.py         ──► models/lstm_P00X.keras
                         data/training_results.json  (MAE/RMSE/MAPE per product)
    │
    ▼
src/stock_recommender.py
    │  Safety Stock → Reorder Point → Recommended Stock → Alerts (6 types)
    ▼
src/inventory.py     ──► data/inventory.json        (persistent stock levels)
    │
    ▼
Streamlit Dashboard
    ├── app.py              (Landing Page)
    ├── pages/1_Overview.py (KPIs, Stock Table, Alerts, Business KPIs)
    ├── pages/2_EDA.py      (8 Charts + SQL Explorer)
    ├── pages/3_Forecasting.py (Forecast + Model Comparison)
    ├── pages/4_Inventory.py   (Editable Stock + Formulas)
    └── pages/5_Reports.py     (CSV / Excel Export)
```

---

## 🗄️ SQL Queries

Sales data is stored in a SQLite database (`data/pharmacy.db`). Example queries:

### 1. Top-Selling Products
```sql
SELECT product_name, SUM(quantity_sold) AS total_sold,
       ROUND(AVG(quantity_sold), 1) AS avg_daily
FROM sales
GROUP BY product_name
ORDER BY total_sold DESC
LIMIT 10;
```

### 2. Monthly Sales Trend
```sql
SELECT SUBSTR(date, 1, 7) AS month,
       SUM(quantity_sold) AS total_units,
       ROUND(SUM(revenue), 2) AS total_revenue
FROM sales
GROUP BY month
ORDER BY month;
```

### 3. Seasonal Demand by Category
```sql
SELECT season, category,
       SUM(quantity_sold) AS total_demand,
       ROUND(AVG(quantity_sold), 1) AS avg_daily
FROM sales
GROUP BY season, category
ORDER BY season, total_demand DESC;
```

### 4. Slow-Moving Products
```sql
SELECT product_name, ROUND(AVG(quantity_sold), 1) AS avg_daily_demand
FROM sales
GROUP BY product_name
HAVING avg_daily_demand < 50
ORDER BY avg_daily_demand ASC;
```

### 5. Year-over-Year Comparison
```sql
SELECT SUBSTR(date, 1, 4) AS year,
       SUM(quantity_sold) AS total_units,
       ROUND(SUM(revenue), 2) AS total_revenue
FROM sales
GROUP BY year ORDER BY year;
```

> 💡 **10 preset queries** are available in the SQL Explorer (EDA page). You can also run custom SQL.

---

## 📈 Model Comparison

Three models are compared per product after training:

| Model          | Type              | Description                                  |
|----------------|-------------------|----------------------------------------------|
| **Naive**      | Baseline          | Predicts last known value for all future days |
| **XGBoost**    | Gradient Boosting | Tabular regressor on 17 engineered features  |
| **LSTM**       | Deep Learning     | Sequential 2-layer LSTM on 30-day windows     |

Results are saved to `data/training_results.json` and displayed per product in the Forecasting page.

---

## 📏 Evaluation Metrics

| Metric               | Formula                                      | Interpretation           |
|----------------------|----------------------------------------------|--------------------------|
| **MAE**              | mean(abs(y_true - y_pred))                   | Avg error in units/day   |
| **RMSE**             | sqrt(mean((y_true - y_pred)²))               | Penalises large errors   |
| **MAPE (%)**         | mean(abs((y_true - y_pred) / y_true)) × 100  | % error (scale-free)     |
| **Directional Acc**  | % of times direction of change is correct    | Trend prediction quality |

---

## 📱 Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 **Landing** | Project overview, setup instructions |
| 🏠 **Overview** | Store KPIs, all-product table, alert panel, business KPIs |
| 📊 **EDA** | 8 charts + interactive SQL Explorer |
| 📈 **Forecasting** | Demand forecast, actual vs predicted, model comparison table |
| 📦 **Inventory** | Editable stock editor, reorder point formulas, alert panel |
| 📥 **Reports** | CSV & Excel exports for forecasts, alerts, and sales data |

---

## 💼 Business Impact

| Problem | Solution | Impact |
|---------|----------|--------|
| Stock-outs on key medicines | Reorder point alerts trigger before depletion | Reduced lost sales |
| Overstocking near expiry | Overstock alerts + expiry risk flag | Reduced wastage |
| Seasonal demand spikes | Seasonal multipliers & buffer adjustment | Better preparedness |
| No visibility of fast vs slow movers | Turnover classification per product | Smarter ordering |
| Manual, intuition-based ordering | Data-driven forecast + recommendation | Process efficiency |

> **"I used SQL to extract and analyze historical sales trends, top-moving products, and seasonal demand patterns before building the forecasting pipeline. I compared LSTM against simpler baselines to validate whether the added complexity was justified, and translated model predictions into actionable inventory KPIs including safety stock, reorder points, and six alert categories."**

---

## 🚀 How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Step 1 — Generate Dataset & Initialise Database
```bash
python generate_dataset.py
python src/db.py
```

### Step 2 — Train Models
```bash
# Train all 12 products (recommended: 30 epochs for demo)
python src/train.py --epochs 30

# Train specific products only
python src/train.py --products P001,P005,P011 --epochs 50
```

### Step 3 — Launch Dashboard
```bash
streamlit run app.py
```

---

## 🔮 Future Scope

| Enhancement | Description |
|-------------|-------------|
| **Real-time billing integration** | Connect to live POS/ERP data instead of CSV |
| **Festival & outbreak signals** | Integrate Google Trends, disease surveillance APIs |
| **Multi-store optimization** | Centralized demand planning across store network |
| **Automatic reorder system** | Generate purchase orders directly from dashboard |
| **Transformer forecasting** | Replace LSTM with Temporal Fusion Transformer (TFT) |
| **Anomaly detection** | Flag unusual demand spikes for manual review |
| **Supplier lead time optimization** | Dynamic lead time learning from historical order data |
| **Expiry date tracking** | Per-batch expiry management integrated with stock levels |

---

## 🛠️ Tech Stack

| Layer         | Technology                          |
|---------------|-------------------------------------|
| ML Framework  | TensorFlow / Keras (LSTM)           |
| Gradient Boost| XGBoost                             |
| Data Layer    | SQLite (via `sqlite3` + `pandas`)   |
| Dashboard     | Streamlit                           |
| Visualization | Plotly                              |
| Data          | Pandas, NumPy, scikit-learn         |
| Export        | openpyxl (Excel), CSV               |

---

*Built as a portfolio project demonstrating end-to-end data science: SQL → EDA → Feature Engineering → Model Comparison → Business Decision Layer.*
