"""
Configuration for Pharmacy Demand Prediction & Stock Analytics System
"""
import os

# ─── Paths ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
SCALER_DIR = os.path.join(BASE_DIR, "scalers")

DATA_FILE = os.path.join(DATA_DIR, "pharmacy_sales.csv")
DB_FILE = os.path.join(DATA_DIR, "pharmacy.db")
INVENTORY_FILE = os.path.join(DATA_DIR, "inventory.json")
TRAINING_RESULTS_FILE = os.path.join(DATA_DIR, "training_results.json")

# ─── Model Hyperparameters ─────────────────────────────────────────────
SEQUENCE_LENGTH = 30          # 30-day lookback window
EPOCHS = 50
BATCH_SIZE = 32
LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 32
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
USE_BIDIRECTIONAL = False     # Set True to use Bidirectional LSTM

# ─── Forecast Horizons ────────────────────────────────────────────────
FORECAST_HORIZONS = [7, 30, 60]

# ─── Indian Seasonal Mapping ──────────────────────────────────────────
SEASON_MAP = {
    1: "Winter",
    2: "Winter",
    3: "Summer",
    4: "Summer",
    5: "Summer",
    6: "Monsoon",
    7: "Monsoon",
    8: "Monsoon",
    9: "Monsoon",
    10: "Post-Monsoon",
    11: "Post-Monsoon",
    12: "Winter",
}

SEASON_COLORS = {
    "Summer": "#FF6B35",
    "Monsoon": "#1B98E0",
    "Post-Monsoon": "#F4A261",
    "Winter": "#2EC4B6",
}

# ─── Indian Festival / Holiday Months ────────────────────────────────
# Months with major Indian festivals that affect pharmacy demand
INDIAN_FESTIVALS = {
    1: "Makar Sankranti",
    3: "Holi",
    4: "Baisakhi",
    8: "Independence Day / Rakshabandhan",
    9: "Ganesh Chaturthi",
    10: "Navratri / Dussehra",
    11: "Diwali / Dhanteras",
    12: "Christmas",
}

# Months where festival activity increases pharmacy traffic
FESTIVAL_MONTHS = {1, 3, 8, 9, 10, 11}

# Peak disease season by category
# (season, category) pairs where disease risk is high
PEAK_DISEASE_SEASONS = {
    ("Monsoon", "fever"),
    ("Monsoon", "antibiotic"),
    ("Monsoon", "hydration"),
    ("Winter", "respiratory"),
    ("Winter", "fever"),
    ("Summer", "hydration"),
    ("Post-Monsoon", "allergy"),
}

# ─── Product Catalog ──────────────────────────────────────────────────
PRODUCTS = {
    "P001": {"name": "Paracetamol 500mg",    "category": "fever",       "base_demand": 120, "unit_price": 15.0,  "lead_time_days": 3},
    "P002": {"name": "ORS Powder",            "category": "hydration",   "base_demand": 80,  "unit_price": 12.0,  "lead_time_days": 4},
    "P003": {"name": "Cough Syrup 100ml",     "category": "respiratory", "base_demand": 60,  "unit_price": 85.0,  "lead_time_days": 5},
    "P004": {"name": "Cetirizine 10mg",       "category": "allergy",     "base_demand": 70,  "unit_price": 25.0,  "lead_time_days": 4},
    "P005": {"name": "Dolo 650",              "category": "fever",       "base_demand": 110, "unit_price": 30.0,  "lead_time_days": 3},
    "P006": {"name": "Azithromycin 500mg",    "category": "antibiotic",  "base_demand": 40,  "unit_price": 120.0, "lead_time_days": 6},
    "P007": {"name": "Vitamin C 500mg",       "category": "supplement",  "base_demand": 90,  "unit_price": 45.0,  "lead_time_days": 5},
    "P008": {"name": "Ibuprofen 400mg",       "category": "painkiller",  "base_demand": 55,  "unit_price": 20.0,  "lead_time_days": 4},
    "P009": {"name": "Amoxicillin 250mg",     "category": "antibiotic",  "base_demand": 45,  "unit_price": 65.0,  "lead_time_days": 6},
    "P010": {"name": "Electral Powder",       "category": "hydration",   "base_demand": 75,  "unit_price": 18.0,  "lead_time_days": 4},
    "P011": {"name": "Insulin Glargine",      "category": "diabetes",    "base_demand": 30,  "unit_price": 450.0, "lead_time_days": 7},
    "P012": {"name": "Antihistamine Tabs",    "category": "allergy",     "base_demand": 65,  "unit_price": 35.0,  "lead_time_days": 4},
}

# ─── Seasonal Demand Multipliers ──────────────────────────────────────
SEASONAL_MULTIPLIERS = {
    ("Monsoon", "fever"):       1.8,
    ("Monsoon", "hydration"):   1.6,
    ("Monsoon", "antibiotic"):  1.5,
    ("Monsoon", "allergy"):     1.3,
    ("Winter", "respiratory"):  1.7,
    ("Winter", "fever"):        1.4,
    ("Winter", "supplement"):   1.3,
    ("Summer", "hydration"):    1.9,
    ("Summer", "painkiller"):   1.2,
    ("Post-Monsoon", "allergy"):1.4,
}

# ─── Stock Recommendation ─────────────────────────────────────────────
# Season → base buffer percentage (added on top of safety stock)
STOCK_BUFFER = {
    "Summer":       0.18,
    "Monsoon":      0.25,
    "Post-Monsoon": 0.15,
    "Winter":       0.20,
}

# Default lead time if not specified per product (days)
DEFAULT_LEAD_TIME = 5

# Safety stock multiplier: Safety Stock = avg_daily * lead_time * factor
SAFETY_STOCK_FACTOR = 0.5

# ─── Alert Thresholds ─────────────────────────────────────────────────
LOW_STOCK_THRESHOLD  = 0.8    # alert if current < 80% of recommended
OVERSTOCK_THRESHOLD  = 1.3    # alert if current > 130% of recommended
CRITICAL_THRESHOLD   = 0.5    # critical if current < 50% of recommended

# Turnover thresholds (units/day) for product classification
FAST_MOVING_THRESHOLD = 90    # avg daily > 90 → fast-moving
SLOW_MOVING_THRESHOLD = 45    # avg daily < 45 → slow-moving

# ─── Store Info ────────────────────────────────────────────────────────
STORE_ID = "STORE_001"
STORE_NAME = "MediCare Pharmacy"
