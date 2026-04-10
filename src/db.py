"""
SQLite database layer for PharmaCast.

Loads the pharmacy sales CSV into a SQLite database and exposes
query helpers for EDA, reporting, and the SQL Explorer dashboard.

Usage (CLI):
    python src/db.py          # initialise / re-load the DB
"""
import os
import sys
import sqlite3
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import DB_FILE, DATA_FILE


# ─── Preset SQL queries shown in the EDA SQL Explorer ─────────────────
PRESET_QUERIES = {
    "Top 10 Products by Total Sales": """
SELECT
    product_name,
    SUM(quantity_sold) AS total_sold,
    ROUND(AVG(quantity_sold), 1) AS avg_daily
FROM sales
GROUP BY product_name
ORDER BY total_sold DESC
LIMIT 10;
""".strip(),

    "Monthly Sales Trend": """
SELECT
    SUBSTR(date, 1, 7)         AS month,
    SUM(quantity_sold)         AS total_sales,
    ROUND(SUM(revenue), 2)     AS total_revenue
FROM sales
GROUP BY month
ORDER BY month;
""".strip(),

    "Seasonal Demand by Category": """
SELECT
    season,
    category,
    SUM(quantity_sold)         AS total_demand,
    ROUND(AVG(quantity_sold), 1) AS avg_daily
FROM sales
GROUP BY season, category
ORDER BY season, total_demand DESC;
""".strip(),

    "Slow-Moving Products (Bottom 5 Avg Daily)": """
SELECT
    product_name,
    category,
    ROUND(AVG(quantity_sold), 1) AS avg_daily_demand,
    SUM(quantity_sold)           AS total_sold
FROM sales
GROUP BY product_name, category
ORDER BY avg_daily_demand ASC
LIMIT 5;
""".strip(),

    "Fast-Moving Products (Top 5 Avg Daily)": """
SELECT
    product_name,
    category,
    ROUND(AVG(quantity_sold), 1) AS avg_daily_demand,
    SUM(quantity_sold)           AS total_sold
FROM sales
GROUP BY product_name, category
ORDER BY avg_daily_demand DESC
LIMIT 5;
""".strip(),

    "Season-wise Revenue": """
SELECT
    season,
    ROUND(SUM(revenue), 2)        AS total_revenue,
    ROUND(AVG(revenue), 2)        AS avg_daily_revenue,
    SUM(quantity_sold)            AS total_units
FROM sales
GROUP BY season
ORDER BY total_revenue DESC;
""".strip(),

    "Expiry Risk Products (Current High Stock)": """
SELECT
    product_name,
    category,
    SUM(expiry_risk)    AS expiry_risk_days,
    ROUND(AVG(current_stock), 0) AS avg_stock
FROM sales
WHERE expiry_risk = 1
GROUP BY product_name, category
ORDER BY expiry_risk_days DESC;
""".strip(),

    "Festival Month vs Normal Month Demand": """
SELECT
    product_name,
    is_festival_month,
    ROUND(AVG(quantity_sold), 1)  AS avg_daily_demand
FROM sales
GROUP BY product_name, is_festival_month
ORDER BY product_name, is_festival_month;
""".strip(),

    "Year-over-Year Sales Comparison": """
SELECT
    SUBSTR(date, 1, 4) AS year,
    SUM(quantity_sold)         AS total_units,
    ROUND(SUM(revenue), 2)     AS total_revenue
FROM sales
GROUP BY year
ORDER BY year;
""".strip(),

    "Low Stock Days per Product": """
SELECT
    product_name,
    COUNT(*)  AS low_stock_days
FROM sales
WHERE current_stock < (
    SELECT AVG(quantity_sold) * 7
    FROM sales s2
    WHERE s2.product_id = sales.product_id
)
GROUP BY product_name
ORDER BY low_stock_days DESC;
""".strip(),
}


# ─── Database initialisation ───────────────────────────────────────────

def get_connection() -> sqlite3.Connection:
    """Return a sqlite3 connection to the pharmacy database."""
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
    return sqlite3.connect(DB_FILE)


def init_db(force_reload: bool = False) -> None:
    """
    Create the `sales` table and load data from the CSV file.

    Args:
        force_reload: If True, drop and recreate the table even if it exists.
    """
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"CSV not found: {DATA_FILE}\n"
            "Run `python generate_dataset.py` first."
        )

    conn = get_connection()
    cur  = conn.cursor()

    if force_reload:
        cur.execute("DROP TABLE IF EXISTS sales")
        conn.commit()

    # Check if table already populated
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='sales'"
    )
    table_exists = cur.fetchone() is not None

    if table_exists and not force_reload:
        cur.execute("SELECT COUNT(*) FROM sales")
        count = cur.fetchone()[0]
        if count > 0:
            print(f"ℹ️  Database already loaded ({count:,} rows). Use force_reload=True to refresh.")
            conn.close()
            return

    # Create table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sales (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            date             TEXT    NOT NULL,
            store_id         TEXT,
            product_id       TEXT    NOT NULL,
            product_name     TEXT,
            category         TEXT,
            season           TEXT,
            quantity_sold    INTEGER NOT NULL,
            unit_price       REAL,
            revenue          REAL,
            current_stock    INTEGER,
            lead_time_days   INTEGER,
            expiry_risk      INTEGER,
            is_festival_month INTEGER
        )
    """)

    # Create indexes for common query patterns
    cur.execute("CREATE INDEX IF NOT EXISTS idx_date        ON sales(date)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_product_id  ON sales(product_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_season      ON sales(season)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_category    ON sales(category)")
    conn.commit()

    # Load CSV
    df = pd.read_csv(DATA_FILE)
    df.to_sql("sales", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()

    print(f"✅ Database initialised: {DB_FILE}")
    print(f"   Rows loaded: {len(df):,}")


# ─── Query helpers ─────────────────────────────────────────────────────

def _query(sql: str, params=()) -> pd.DataFrame:
    """Execute a SQL query and return a DataFrame."""
    conn = get_connection()
    try:
        df = pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()
    return df


def query_top_selling(n: int = 10) -> pd.DataFrame:
    """Top N products by total units sold."""
    return _query("""
        SELECT
            product_name,
            category,
            SUM(quantity_sold)           AS total_sold,
            ROUND(AVG(quantity_sold), 1) AS avg_daily,
            ROUND(SUM(revenue), 2)       AS total_revenue
        FROM sales
        GROUP BY product_name, category
        ORDER BY total_sold DESC
        LIMIT ?
    """, (n,))


def query_monthly_trend() -> pd.DataFrame:
    """Monthly sales trend across all products."""
    return _query("""
        SELECT
            SUBSTR(date, 1, 7)         AS month,
            SUM(quantity_sold)         AS total_units,
            ROUND(SUM(revenue), 2)     AS total_revenue,
            COUNT(DISTINCT product_id) AS products_sold
        FROM sales
        GROUP BY month
        ORDER BY month
    """)


def query_seasonal_demand() -> pd.DataFrame:
    """Average daily demand grouped by season and product."""
    return _query("""
        SELECT
            season,
            product_name,
            category,
            SUM(quantity_sold)           AS total_demand,
            ROUND(AVG(quantity_sold), 1) AS avg_daily
        FROM sales
        GROUP BY season, product_name, category
        ORDER BY season, avg_daily DESC
    """)


def query_slow_movers(threshold: float = 50.0) -> pd.DataFrame:
    """Products with average daily demand below threshold."""
    return _query("""
        SELECT
            product_name,
            category,
            ROUND(AVG(quantity_sold), 1) AS avg_daily_demand,
            SUM(quantity_sold)           AS total_sold
        FROM sales
        GROUP BY product_name, category
        HAVING avg_daily_demand < ?
        ORDER BY avg_daily_demand ASC
    """, (threshold,))


def query_product_sales(product_id: str) -> pd.DataFrame:
    """Full time series for a single product."""
    return _query("""
        SELECT date, quantity_sold, revenue, current_stock,
               season, is_festival_month
        FROM   sales
        WHERE  product_id = ?
        ORDER  BY date
    """, (product_id,))


def query_category_revenue() -> pd.DataFrame:
    """Total revenue per category."""
    return _query("""
        SELECT
            category,
            ROUND(SUM(revenue), 2)       AS total_revenue,
            SUM(quantity_sold)           AS total_units,
            ROUND(AVG(quantity_sold), 1) AS avg_daily
        FROM sales
        GROUP BY category
        ORDER BY total_revenue DESC
    """)


def query_yoy_comparison() -> pd.DataFrame:
    """Year-over-year sales summary."""
    return _query("""
        SELECT
            SUBSTR(date, 1, 4)     AS year,
            SUM(quantity_sold)     AS total_units,
            ROUND(SUM(revenue), 2) AS total_revenue
        FROM sales
        GROUP BY year
        ORDER BY year
    """)


def execute_custom(sql: str) -> pd.DataFrame:
    """Execute arbitrary SQL and return result as DataFrame."""
    return _query(sql)


def db_status() -> dict:
    """Return basic stats about the database."""
    conn = get_connection()
    cur  = conn.cursor()
    try:
        cur.execute("SELECT COUNT(*) FROM sales")
        rows = cur.fetchone()[0]
        cur.execute("SELECT MIN(date), MAX(date) FROM sales")
        date_range = cur.fetchone()
        cur.execute("SELECT COUNT(DISTINCT product_id) FROM sales")
        products = cur.fetchone()[0]
    except Exception:
        return {"status": "not_initialised"}
    finally:
        conn.close()

    return {
        "status": "ok",
        "rows": rows,
        "date_from": date_range[0],
        "date_to": date_range[1],
        "products": products,
        "db_path": DB_FILE,
    }


if __name__ == "__main__":
    print("🗄️  Initialising pharmacy SQLite database...")
    init_db(force_reload=True)
    status = db_status()
    print(f"\n📊 DB Status: {status}")
    print("\n🔍 Sample — Top Selling:")
    print(query_top_selling(5).to_string(index=False))
    print("\n📅 Sample — Monthly Trend (last 6 months):")
    print(query_monthly_trend().tail(6).to_string(index=False))
