import csv
import time
from tqdm import tqdm
from config import USE_MOCK_TWITTER
import logging
from datetime import datetime
from sentiment_engine.processor import get_sentiment
from utils.tagging import detect_tags
from sentiment_engine.aggregator import add_to_sentiment_buffer
import random

logger = logging.getLogger("twitter_stream")

REQUIRED_ASSETS = {
    "BTC", "ETH", "DOGE", "SHIBA", "SOL", "XRP",
    "AAPL", "TSLA", "META", "AMZN", "GOOGL", "MSFT", "NFLX",
    "NIFTY", "GSPC", "NASDAQ", "DOWJONES"
}


def stream_twitter(file_path="data/mock_twitter.csv", delay=1.5):
    """
    Streams Twitter data (mock or live, based on config).
    """
    import os
    # logger.info(f"[DEBUG] Attempting to open Twitter file: {file_path} (cwd: {os.getcwd()})")
    if USE_MOCK_TWITTER:
        try:
            with open(file_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in tqdm(reader, desc="Streaming Tweets"):
                    tweet = {
                        "timestamp": datetime.now(),
                        "text": row['tweet_text'],
                        "user_influence": float(row['user_influence'])
                    }
                    # logger.info(f"[DEBUG] Processing tweet: {tweet}")
                    sentiment = {"label": random.choice(["bullish", "bearish", "neutral"]), "score": round(random.uniform(-1, 1), 4)}
                    tags = detect_tags(row['tweet_text'])
                    tags = [t for t in tags if t in REQUIRED_ASSETS]
                    if not tags:
                        continue  # Skip tweets with no required asset tag
                    # logger.info(f"[DEBUG] Sentiment score: {sentiment['score']} | Tags: {tags}")
                    add_to_sentiment_buffer(sentiment["score"], "twitter", tweet["timestamp"], influence=tweet["user_influence"], tags=tags)
                    yield tweet
                    time.sleep(delay)  # simulate real-time
        except Exception as e:
            logger.error(f"[ERROR] Could not open or process mock Twitter file: {e}")
    else:
        logger.warning("Live Twitter API not implemented. Set USE_MOCK_TWITTER=True in config.py.")
        # Placeholder for live Twitter API integration
        return
