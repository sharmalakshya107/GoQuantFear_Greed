import praw
import time
from utils.tagging import detect_tags
from config import REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT, USE_MOCK_REDDIT
import logging
from datetime import datetime
from sentiment_engine.processor import get_sentiment
from sentiment_engine.aggregator import add_to_sentiment_buffer
import random

logger = logging.getLogger("reddit_stream")

reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT
)

REQUIRED_ASSETS = {
    "BTC", "ETH", "DOGE", "SHIBA", "SOL", "XRP",
    "AAPL", "TSLA", "META", "AMZN", "GOOGL", "MSFT", "NFLX",
    "NIFTY", "GSPC", "NASDAQ", "DOWJONES"
}

def stream_reddit_posts(subreddits=["CryptoCurrency", "Bitcoin", "stocks", "wallstreetbets"], limit=10, delay=2):
    """
    Streams Reddit posts from specified subreddits.
    Includes tagging for asset mentions like BTC, ETH, AAPL, etc.
    """
    if USE_MOCK_REDDIT:
        logger.warning("Mock Reddit stream not implemented. Generating random mock posts.")
        for i in range(limit):
            content = f"Mock Reddit post {i} about BTC, ETH, AAPL, etc."
            post = {
                "timestamp": datetime.now(),
                "text": content,
                "subreddit": random.choice(["CryptoCurrency", "Bitcoin", "stocks", "wallstreetbets"]),
                "score": random.randint(0, 100),
                "comments": random.randint(0, 50),
                "tags": detect_tags(content) or ["MARKET"]
            }
            timestamp = post["timestamp"]
            sentiment = {"label": random.choice(["bullish", "bearish", "neutral"]), "score": round(random.uniform(-1, 1), 4)}
            tags = post["tags"]
            tags = [t for t in tags if t in REQUIRED_ASSETS]
            if not tags:
                continue  # Skip posts with no required asset tag
            # logger.info(f"[DEBUG] Sentiment score: {sentiment['score']} | Tags: {tags}")
            add_to_sentiment_buffer(sentiment["score"], "reddit", timestamp, influence=1.2, tags=tags)
            yield post
        return
    seen_ids = set()
    while True:
        for subreddit_name in subreddits:
            try:
                subreddit = reddit.subreddit(subreddit_name)
                for submission in subreddit.new(limit=limit):
                    if submission.id in seen_ids:
                        continue
                    seen_ids.add(submission.id)
                    content = submission.title + " " + submission.selftext
                    post = {
                        "timestamp": datetime.now(),
                        "text": content,
                        "subreddit": subreddit_name,
                        "score": submission.score,
                        "comments": submission.num_comments,
                        "tags": detect_tags(content) or ["MARKET"]
                    }
                    timestamp = post["timestamp"]
                    sentiment = get_sentiment(content)
                    tags = post["tags"]
                    tags = [t for t in tags if t in REQUIRED_ASSETS]
                    if not tags:
                        continue  # Skip posts with no required asset tag
                    # logger.info(f"[DEBUG] Sentiment score: {sentiment['score']} | Tags: {tags}")
                    add_to_sentiment_buffer(sentiment["score"], "reddit", timestamp, influence=1.2, tags=tags)
                    yield post
            except Exception as e:
                logger.error(f"Error streaming Reddit posts from {subreddit_name}: {e}")
        time.sleep(delay)  # avoid being rate-limited
