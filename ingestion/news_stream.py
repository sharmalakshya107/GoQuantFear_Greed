# ingestion/news_stream.py

from datetime import datetime
import feedparser
from sentiment_engine.processor import get_sentiment
from utils.tagging import detect_tags
from sentiment_engine.aggregator import add_to_sentiment_buffer
from config import NEWS_FEEDS
import logging

logger = logging.getLogger("news_stream")

def fetch_news_articles():
    for url in NEWS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]:  # limit to top 5 articles per source
                title = entry.get("title", "")
                summary = entry.get("summary", "")
                link = entry.get("link", "")
                # Use current time for demo/analytics
                timestamp = datetime.now()
                content = f"{title} {summary}"
                sentiment = get_sentiment(content)
                tags = detect_tags(content) or ["MARKET"]
                # print(f"[{timestamp}] 📰 {title}")
                # print(f"   🔗 {link}")
                # print(f"🧠 Sentiment: {sentiment['label']} ({sentiment['score']})")
                # print(f"🏷️ Tags: {tags}")
                # print("-" * 60)
                add_to_sentiment_buffer(sentiment["score"], "news", timestamp, influence=1.5, tags=tags)
        except Exception as e:
            logger.error(f"Error fetching news from {url}: {e}")
