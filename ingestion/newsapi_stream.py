import requests
import time
from config import NEWSAPI_KEY, ENABLE_NEWSAPI
from sentiment_engine.aggregator import add_to_sentiment_buffer
from utils.tagging import detect_tags
from datetime import datetime

NEWSAPI_URL = "https://newsapi.org/v2/top-headlines"

# Streams latest news headlines (finance category)
def stream_newsapi_headlines(country="us", category="business", delay=60):
    if not ENABLE_NEWSAPI:
        print("[NewsAPI] Integration disabled in config.")
        return
    print("[NewsAPI] Streaming latest news headlines...")
    seen_titles = set()
    while True:
        params = {
            "apiKey": NEWSAPI_KEY,
            "country": country,
            "category": category,
            "pageSize": 10
        }
        try:
            resp = requests.get(NEWSAPI_URL, params=params, timeout=10)
            data = resp.json()
            if data.get("status") == "ok":
                for article in data.get("articles", []):
                    title = article.get("title", "")
                    if title in seen_titles:
                        continue
                    seen_titles.add(title)
                    summary = article.get("description", "")
                    content = f"{title} {summary}"
                    tags = detect_tags(content)
                    add_to_sentiment_buffer(
                        score=0.0,  # Neutral for news headline only
                        source="newsapi",
                        timestamp=datetime.now(),
                        influence=1.0,
                        tags=tags
                    )
                    print(f"[NewsAPI] {title}")
            else:
                print(f"[NewsAPI] Error: {data}")
        except Exception as e:
            print(f"[NewsAPI] Error fetching news: {e}")
        time.sleep(delay) 