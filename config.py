"""
Central configuration for GoQuant Fear & Greed Sentiment Engine
Edit this file to change data sources, API keys, analysis parameters, and more.
NOTE: All API keys below are placeholders for interview submission. Insert your own credentials to run the project.
"""

# --- Data Source Toggles ---
USE_MOCK_TWITTER = True  # Set to False to use real Twitter API (if implemented)
USE_MOCK_REDDIT = False  # Set to True to use mock Reddit data (if available)

# --- API Keys & Endpoints ---
ALPHA_VANTAGE_API_KEY = "YOUR_ALPHA_VANTAGE_API_KEY"
NEWSAPI_KEY = "YOUR_NEWSAPI_KEY"
ALTERNATIVE_ME_API_URL = "https://api.alternative.me/fng/"
STOCKTWITS_CLIENT_ID = "YOUR_STOCKTWITS_CLIENT_ID"
STOCKTWITS_CLIENT_SECRET = "YOUR_STOCKTWITS_CLIENT_SECRET"
ALPACA_API_KEY = "YOUR_ALPACA_API_KEY"
ALPACA_API_SECRET = "YOUR_ALPACA_API_SECRET"
BINANCE_API_KEY = "YOUR_BINANCE_API_KEY"
BINANCE_API_SECRET = "YOUR_BINANCE_API_SECRET"
TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_TELEGRAM_CHAT_ID"
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
SLACK_WEBHOOK_URL = "YOUR_SLACK_WEBHOOK_URL"

# --- Reddit API (add these for reddit_stream.py) ---
REDDIT_CLIENT_ID = "YOUR_REDDIT_CLIENT_ID"
REDDIT_CLIENT_SECRET = "YOUR_REDDIT_CLIENT_SECRET"
REDDIT_USER_AGENT = "python:goquant-fear-greed:1.0 (by /u/yourusername)"
# --- API Toggles ---
ENABLE_ALPHA_VANTAGE = True
ENABLE_NEWSAPI = True
ENABLE_ALTERNATIVE_ME = True
ENABLE_STOCKTWITS = False
ENABLE_ALPACA = True
ENABLE_BINANCE = False
ENABLE_TELEGRAM = True
ENABLE_OPENAI = True
ENABLE_SLACK = True

# --- NLP Model Selection ---
NLP_MODEL = "vader"  # Options: "vader", "finbert", "spacy"

# --- Sentiment Aggregation ---
BUFFER_WINDOW_MINUTES = 5
ROLLING_N = 30
HALF_LIFE_SEC = 1800

# --- Assets & Feeds ---
ASSETS = {
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "DOGE-USD": "Dogecoin",
    "SHIBA-USD": "Shiba Inu",
    "SOL-USD": "Solana",
    "XRP-USD": "Ripple",
    "AAPL": "Apple",
    "TSLA": "Tesla",
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "GOOGL": "Alphabet",
    "META": "Meta Platforms",
    "NFLX": "Netflix",
    "^GSPC": "S&P 500",
    "^NSEI": "Nifty 50",
    "^DJI": "Dow Jones",
    "^IXIC": "NASDAQ",
    "^RUT": "Russell 2000"
}
# ASSETS should include all tickers you want to track.
# Use yfinance-compatible symbols for stocks/indices (e.g., 'AAPL', 'TSLA', '^GSPC')
# For crypto, use symbols like 'BTC-USD', 'ETH-USD', etc. for CoinGecko fallback.
NEWS_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://www.bloomberg.com/feed/podcast/etf-report.xml"
]

# --- Visualization ---
ENABLE_VISUALIZATION = True
VISUALIZATION_REFRESH_SEC = 10

# --- Logging ---
LOG_LEVEL = "INFO"  # Options: "DEBUG", "INFO", "WARNING", "ERROR" 

# --- Multi-Language Sentiment (Future) ---
MULTI_LANGUAGE_SUPPORT = False 