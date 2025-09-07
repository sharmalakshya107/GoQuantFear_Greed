import requests
import time
from config import STOCKTWITS_CLIENT_ID, STOCKTWITS_CLIENT_SECRET, ENABLE_STOCKTWITS, ASSETS
from sentiment_engine.aggregator import add_to_sentiment_buffer
from utils.tagging import detect_tags
from datetime import datetime

STOCKTWITS_URL = "https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"

def stream_stocktwits(delay=60):
    if not ENABLE_STOCKTWITS:
        print("[StockTwits] Integration disabled in config.")
        return
    print("[StockTwits] Streaming latest messages...")
    seen_ids = set()
    for symbol in ASSETS.keys():
        while True:
            try:
                url = STOCKTWITS_URL.format(symbol=symbol)
                resp = requests.get(url, timeout=10)
                data = resp.json()
                if "messages" in data:
                    for msg in data["messages"]:
                        msg_id = msg["id"]
                        if msg_id in seen_ids:
                            continue
                        seen_ids.add(msg_id)
                        body = msg.get("body", "")
                        sentiment = msg.get("entities", {}).get("sentiment", {}).get("basic", "None")
                        tags = detect_tags(body)
                        score = 0.0
                        if sentiment == "Bullish":
                            score = 1.0
                        elif sentiment == "Bearish":
                            score = -1.0
                        add_to_sentiment_buffer(
                            score=score,
                            source="stocktwits",
                            timestamp=datetime.strptime(msg["created_at"], "%Y-%m-%dT%H:%M:%SZ"),
                            influence=1.0,
                            tags=tags
                        )
                        print(f"[StockTwits] {symbol}: {sentiment} | {body}")
                else:
                    print(f"[StockTwits] Unexpected response: {data}")
            except Exception as e:
                print(f"[StockTwits] Error fetching {symbol}: {e}")
            time.sleep(delay) 