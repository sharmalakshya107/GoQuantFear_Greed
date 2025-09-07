import requests
from config import ALPACA_API_KEY, ALPACA_API_SECRET, ENABLE_ALPACA
from datetime import datetime

ALPACA_BASE_URL = "https://paper-api.alpaca.markets/v2"
HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_API_SECRET
}

def fetch_alpaca_account():
    if not ENABLE_ALPACA:
        print("[Alpaca] Integration disabled in config.")
        return None
    try:
        resp = requests.get(f"{ALPACA_BASE_URL}/account", headers=HEADERS, timeout=10)
        return resp.json()
    except Exception as e:
        print(f"[Alpaca] Error fetching account: {e}")
        return None

def fetch_alpaca_orders(status="all", limit=10):
    if not ENABLE_ALPACA:
        return []
    try:
        resp = requests.get(f"{ALPACA_BASE_URL}/orders", headers=HEADERS, params={"status": status, "limit": limit}, timeout=10)
        return resp.json()
    except Exception as e:
        print(f"[Alpaca] Error fetching orders: {e}")
        return []

def submit_alpaca_order(symbol, qty, side, type="market", time_in_force="gtc", demo=True):
    if not ENABLE_ALPACA or demo:
        print(f"[Alpaca] Demo mode: {side} {qty} {symbol} (no real trade)")
        return {"status": "demo", "symbol": symbol, "qty": qty, "side": side}
    try:
        order = {
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "type": type,
            "time_in_force": time_in_force
        }
        resp = requests.post(f"{ALPACA_BASE_URL}/orders", headers=HEADERS, json=order, timeout=10)
        return resp.json()
    except Exception as e:
        print(f"[Alpaca] Error submitting order: {e}")
        return {"status": "error", "error": str(e)} 