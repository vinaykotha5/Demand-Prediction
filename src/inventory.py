"""
Persistent inventory management for PharmaCast.
Reads/writes current stock levels to data/inventory.json.
"""
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import INVENTORY_FILE, PRODUCTS, DATA_DIR


def _default_inventory() -> dict:
    """Generate sensible starting stock levels from product catalog."""
    return {
        pid: {
            "current_stock": info["base_demand"] * 30,  # 30-day buffer as default
            "last_updated": "never",
        }
        for pid, info in PRODUCTS.items()
    }


def load_inventory() -> dict:
    """
    Load inventory from JSON file.
    If the file doesn't exist, create it with default stock levels.

    Returns:
        dict: {product_id: {"current_stock": int, "last_updated": str}}
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(INVENTORY_FILE):
        data = _default_inventory()
        save_inventory(data)
        return data

    with open(INVENTORY_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Back-fill any new products not yet in the file
    for pid in PRODUCTS:
        if pid not in data:
            data[pid] = {"current_stock": PRODUCTS[pid]["base_demand"] * 30, "last_updated": "never"}

    return data


def save_inventory(data: dict) -> None:
    """
    Persist inventory to JSON file.

    Args:
        data: {product_id: {"current_stock": int, "last_updated": str}}
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(INVENTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_stock(product_id: str) -> int:
    """Return current stock level for a single product."""
    inv = load_inventory()
    return inv.get(product_id, {}).get("current_stock", 0)


def update_stock(product_id: str, new_quantity: int, timestamp: str = None) -> None:
    """
    Set the current stock for a product.

    Args:
        product_id: e.g. "P001"
        new_quantity: absolute stock count
        timestamp: ISO string; defaults to now
    """
    from datetime import datetime
    if timestamp is None:
        timestamp = datetime.now().isoformat(timespec="seconds")

    inv = load_inventory()
    if product_id not in inv:
        inv[product_id] = {}
    inv[product_id]["current_stock"] = max(0, int(new_quantity))
    inv[product_id]["last_updated"]  = timestamp
    save_inventory(inv)


def bulk_update(stock_dict: dict) -> None:
    """
    Update multiple products at once.

    Args:
        stock_dict: {product_id: new_quantity}
    """
    from datetime import datetime
    ts  = datetime.now().isoformat(timespec="seconds")
    inv = load_inventory()
    for pid, qty in stock_dict.items():
        if pid not in inv:
            inv[pid] = {}
        inv[pid]["current_stock"] = max(0, int(qty))
        inv[pid]["last_updated"]  = ts
    save_inventory(inv)


def get_all_stock_levels() -> dict:
    """Return {product_id: current_stock} for all products."""
    inv = load_inventory()
    return {pid: v["current_stock"] for pid, v in inv.items()}


def reset_to_defaults() -> None:
    """Reset all stock levels to the default 30-day buffer."""
    save_inventory(_default_inventory())
    print("🔁 Inventory reset to defaults.")


if __name__ == "__main__":
    print("Current inventory:")
    for pid, info in load_inventory().items():
        name = PRODUCTS.get(pid, {}).get("name", pid)
        print(f"  {pid} | {name:25s} | Stock: {info['current_stock']:,} | Updated: {info['last_updated']}")
