import requests
from config import BINANCE_API_KEY, BINANCE_API_SECRET, ENABLE_BINANCE
from datetime import datetime

BINANCE_BASE_URL = "https://api.binance.com/api/v3"
HEADERS = {
    "X-MBX-APIKEY": BINANCE_API_KEY
}

def fetch_binance_account():
    if not ENABLE_BINANCE:
        print("[Binance] Integration disabled in config.")
        return None
    # Demo: Return mock account info
    return {"account": "demo", "timestamp": datetime.now().isoformat()}

def fetch_binance_trades(symbol, limit=10):
    if not ENABLE_BINANCE:
        return []
    try:
        resp = requests.get(f"{BINANCE_BASE_URL}/trades", headers=HEADERS, params={"symbol": symbol, "limit": limit}, timeout=10)
        return resp.json()
    except Exception as e:
        print(f"[Binance] Error fetching trades: {e}")
        return []

def submit_binance_order(symbol, qty, side, type="MARKET", demo=True):
    if not ENABLE_BINANCE or demo:
        print(f"[Binance] Demo mode: {side} {qty} {symbol} (no real trade)")
        return {"status": "demo", "symbol": symbol, "qty": qty, "side": side}
    # Real trading not implemented for safety
    return {"status": "error", "error": "Real trading not implemented"} 