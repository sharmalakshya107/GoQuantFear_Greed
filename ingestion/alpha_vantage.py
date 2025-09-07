import requests
import time
from config import ALPHA_VANTAGE_API_KEY, ENABLE_ALPHA_VANTAGE, ASSETS
from sentiment_engine.aggregator import add_to_sentiment_buffer
from utils.tagging import detect_tags
from datetime import datetime

ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"

# Supported function: TIME_SERIES_INTRADAY (stocks), CRYPTO_INTRADAY (crypto)
def stream_alpha_vantage_prices(interval="1min", delay=60):
    if not ENABLE_ALPHA_VANTAGE:
        print("[AlphaVantage] Integration disabled in config.")
        return
    print("[AlphaVantage] Streaming live prices...")
    while True:
        for symbol in ASSETS.keys():
            params = {
                "function": "TIME_SERIES_INTRADAY",
                "symbol": symbol,
                "interval": interval,
                "apikey": ALPHA_VANTAGE_API_KEY
            }
            try:
                resp = requests.get(ALPHA_VANTAGE_URL, params=params, timeout=10)
                data = resp.json()
                ts_key = f"Time Series ({interval})"
                if ts_key in data:
                    latest_time = sorted(data[ts_key].keys())[-1]
                    price = float(data[ts_key][latest_time]["4. close"])
                    # Normalize and add to buffer
                    text = f"{symbol} price update: {price} at {latest_time}"
                    tags = detect_tags(text)
                    add_to_sentiment_buffer(
                        score=0.0,  # Neutral for price-only
                        source="alpha_vantage",
                        timestamp=datetime.now(),
                        influence=1.0,
                        tags=tags
                    )
                    print(f"[AlphaVantage] {symbol}: {price} @ {latest_time}")
                else:
                    print(f"[AlphaVantage] No data for {symbol}: {data.get('Note') or data}")
            except Exception as e:
                print(f"[AlphaVantage] Error fetching {symbol}: {e}")
            time.sleep(12)  # Alpha Vantage free tier rate limit
        time.sleep(delay) 