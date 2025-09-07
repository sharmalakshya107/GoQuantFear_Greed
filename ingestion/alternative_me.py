import requests
import time
from config import ALTERNATIVE_ME_API_URL, ENABLE_ALTERNATIVE_ME
from sentiment_engine.aggregator import add_to_sentiment_buffer
from datetime import datetime

def stream_alternative_me_fgi(delay=300):
    if not ENABLE_ALTERNATIVE_ME:
        print("[Alternative.me] Integration disabled in config.")
        return
    print("[Alternative.me] Streaming Fear & Greed Index...")
    while True:
        try:
            resp = requests.get(ALTERNATIVE_ME_API_URL, timeout=10)
            data = resp.json()
            if data.get("name") == "Fear and Greed Index" and data.get("data"):
                latest = data["data"][0]
                value = float(latest["value"])
                value_classification = latest["value_classification"]
                timestamp = datetime.fromtimestamp(int(latest["timestamp"]))
                add_to_sentiment_buffer(
                    score=value / 100.0 * 2 - 1,  # Normalize 0-100 to -1 to 1
                    source="alternative_me",
                    timestamp=timestamp,
                    influence=2.0,
                    tags=[value_classification.lower()]
                )
                print(f"[Alternative.me] FGI: {value} ({value_classification}) @ {timestamp}")
            else:
                print(f"[Alternative.me] Unexpected response: {data}")
        except Exception as e:
            print(f"[Alternative.me] Error fetching FGI: {e}")
        time.sleep(delay) 