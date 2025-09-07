import threading
import logging
from ingestion.twitter_stream import stream_twitter
from ingestion.reddit_stream import stream_reddit_posts
from ingestion.news_stream import fetch_news_articles
from ingestion.finance_data import fetch_price_data
from ingestion.fund_flow import stream_fund_flows
from ingestion.economic_indicators import stream_economic_indicators
from ingestion.alpha_vantage import stream_alpha_vantage_prices
from ingestion.newsapi_stream import stream_newsapi_headlines
from ingestion.alternative_me import stream_alternative_me_fgi
from ingestion.stocktwits_stream import stream_stocktwits
from ingestion.alpaca_stream import fetch_alpaca_account, fetch_alpaca_orders, submit_alpaca_order
from ingestion.binance_stream import fetch_binance_account, fetch_binance_trades, submit_binance_order
from ingestion.telegram_bot import send_telegram_message
from ingestion.slack_bot import send_slack_message
from nlp.openai_sentiment import get_openai_sentiment
from sentiment_engine.processor import get_sentiment as analyze_sentiment
from sentiment_engine.aggregator import add_to_sentiment_buffer, compute_scores
from signal_engine.generator import generate_signals, highlight_signal_changes, write_signals_to_csv
from utils.tagging import detect_tags
from config import ENABLE_VISUALIZATION, LOG_LEVEL, ENABLE_ALPHA_VANTAGE, ENABLE_NEWSAPI, ENABLE_ALTERNATIVE_ME, ENABLE_STOCKTWITS, ENABLE_ALPACA, ENABLE_BINANCE, ENABLE_TELEGRAM, ENABLE_SLACK, ENABLE_OPENAI
import time
from risk.portfolio import get_portfolio_risk
from datetime import datetime
import os

# Setup logging
if not os.path.exists('logs'):
    os.makedirs('logs')
file_handler = logging.FileHandler('logs/ingestion_debug.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
logging.getLogger().addHandler(file_handler)

fund_flow_data = []
economic_data = []

# --- Global Data Source Status ---
data_source_status = {
    'twitter': 'pending',
    'reddit': 'pending',
    'news': 'pending',
    'finance': 'pending',
    'fund_flow': 'pending',
    'economic_indicators': 'pending',
    'alpha_vantage': 'pending',
    'newsapi': 'pending',
    'alternative_me': 'pending',
    'stocktwits': 'pending',
    'alpaca': 'pending',
    'binance': 'pending',
    'telegram': 'pending',
    'slack': 'pending',
}

def set_status(source, status):
    data_source_status[source] = status
    logging.info(f"[STATUS] {source}: {status}")

def get_data_source_status():
    return data_source_status.copy()

# --- Ingestion Threads ---
def run_twitter():
    set_status('twitter', 'running')
    try:
        logging.info("\n🚀 Streaming Twitter Feed:\n")
        for tweet in stream_twitter():
            if ENABLE_OPENAI:
                sentiment = get_openai_sentiment(tweet["text"])
            else:
                sentiment = analyze_sentiment(tweet["text"])
            tags = detect_tags(tweet["text"])
            logging.info(f"[{tweet['timestamp']}] {tweet['text']} (Influence: {tweet['user_influence']})")
            logging.info(f"🧠 Sentiment: {sentiment['label']} ({sentiment['score']})")
            logging.info(f"🏷️ Tags: {tags}")
            logging.info("-" * 60)
            add_to_sentiment_buffer(
                sentiment["score"],
                "twitter",
                datetime.now(),
                influence=tweet["user_influence"],
                tags=tags
            )
        set_status('twitter', 'finished')
    except Exception as e:
        set_status('twitter', f'error: {e}')
        logging.error(f"[ERROR] Twitter ingestion failed: {e}")

def run_reddit():
    set_status('reddit', 'running')
    try:
        logging.info("\n🚀 Reddit Live Stream:\n")
        for post in stream_reddit_posts(limit=15):
            if ENABLE_OPENAI:
                sentiment = get_openai_sentiment(post["text"])
            else:
                sentiment = analyze_sentiment(post["text"])
            tags = detect_tags(post["text"])
            logging.info(f"[{post['timestamp']}] r/{post['subreddit']}")
            logging.info(f"🧵 {post['text'][:100]}...")
            logging.info(f"💬 {post['comments']} comments | 👍 {post['score']} votes")
            logging.info(f"🧠 Sentiment: {sentiment['label']} ({sentiment['score']})")
            logging.info(f"🏷️ Tags: {tags}")
            logging.info("-" * 60)
            add_to_sentiment_buffer(
                sentiment["score"],
                "reddit",
                datetime.now(),
                influence=1.2,
                tags=tags
            )
        set_status('reddit', 'finished')
    except Exception as e:
        set_status('reddit', f'error: {e}')
        logging.error(f"[ERROR] Reddit ingestion failed: {e}")

def run_news():
    set_status('news', 'running')
    try:
        print("\n📰 News Feed Stream:\n")
        fetch_news_articles()
        set_status('news', 'finished')
    except Exception as e:
        set_status('news', f'error: {e}')
        logging.error(f"[ERROR] News ingestion failed: {e}")

def run_finance():
    set_status('finance', 'running')
    try:
        print("\n📊 Fetching Market Prices:\n")
        fetch_price_data()
        set_status('finance', 'finished')
    except Exception as e:
        set_status('finance', f'error: {e}')
        logging.error(f"[ERROR] Finance ingestion failed: {e}")

def run_fund_flow():
    set_status('fund_flow', 'running')
    global fund_flow_data
    try:
        logging.info("\n💸 Streaming Fund Flow Data:\n")
        for flow in stream_fund_flows():
            fund_flow_data.append(flow)
            logging.info(f"[{flow['timestamp']}] Fund Flows: {flow['flows']}")
            if len(fund_flow_data) > 100:
                fund_flow_data = fund_flow_data[-100:]
        set_status('fund_flow', 'finished')
    except Exception as e:
        set_status('fund_flow', f'error: {e}')
        logging.error(f"[ERROR] Fund flow ingestion failed: {e}")

def run_economic_indicators():
    set_status('economic_indicators', 'running')
    global economic_data
    try:
        logging.info("\n📈 Streaming Economic Indicators:\n")
        for econ in stream_economic_indicators():
            economic_data.append(econ)
            logging.info(f"[{econ['timestamp']}] Economic Data: {econ}")
            if len(economic_data) > 100:
                economic_data = economic_data[-100:]
        set_status('economic_indicators', 'finished')
    except Exception as e:
        set_status('economic_indicators', f'error: {e}')
        logging.error(f"[ERROR] Economic indicators ingestion failed: {e}")

# --- New API Ingestion Threads ---
def run_alpha_vantage():
    if ENABLE_ALPHA_VANTAGE:
        set_status('alpha_vantage', 'running')
        try:
            print("\n🔗 Alpha Vantage Live Prices:\n")
            stream_alpha_vantage_prices()
            set_status('alpha_vantage', 'finished')
        except Exception as e:
            set_status('alpha_vantage', f'error: {e}')
            logging.error(f"[ERROR] Alpha Vantage ingestion failed: {e}")

def run_newsapi():
    if ENABLE_NEWSAPI:
        set_status('newsapi', 'running')
        try:
            print("\n📰 NewsAPI Headlines:\n")
            stream_newsapi_headlines()
            set_status('newsapi', 'finished')
        except Exception as e:
            set_status('newsapi', f'error: {e}')
            logging.error(f"[ERROR] NewsAPI ingestion failed: {e}")

def run_alternative_me():
    if ENABLE_ALTERNATIVE_ME:
        set_status('alternative_me', 'running')
        try:
            print("\n🟢 Alternative.me Fear & Greed Index:\n")
            stream_alternative_me_fgi()
            set_status('alternative_me', 'finished')
        except Exception as e:
            set_status('alternative_me', f'error: {e}')
            logging.error(f"[ERROR] Alternative.me ingestion failed: {e}")

def run_stocktwits():
    if ENABLE_STOCKTWITS:
        set_status('stocktwits', 'running')
        try:
            print("\n💬 StockTwits Social Sentiment:\n")
            stream_stocktwits()
            set_status('stocktwits', 'finished')
        except Exception as e:
            set_status('stocktwits', f'error: {e}')
            logging.error(f"[ERROR] StockTwits ingestion failed: {e}")

# --- Trading/Notification Threads ---
def run_alpaca_demo():
    if ENABLE_ALPACA:
        set_status('alpaca', 'running')
        try:
            print("\n🟠 Alpaca Paper Trading (Demo):\n")
            acct = fetch_alpaca_account()
            print(f"[Alpaca] Account: {acct}")
            orders = fetch_alpaca_orders()
            print(f"[Alpaca] Recent Orders: {orders}")
            submit_alpaca_order("AAPL", 1, "buy", demo=True)
            set_status('alpaca', 'finished')
        except Exception as e:
            set_status('alpaca', f'error: {e}')
            logging.error(f"[ERROR] Alpaca demo failed: {e}")

def run_binance_demo():
    if ENABLE_BINANCE:
        set_status('binance', 'running')
        try:
            print("\n🟡 Binance Demo Trading:\n")
            acct = fetch_binance_account()
            print(f"[Binance] Account: {acct}")
            trades = fetch_binance_trades("BTCUSDT")
            print(f"[Binance] Recent Trades: {trades}")
            submit_binance_order("BTCUSDT", 0.01, "buy", demo=True)
            set_status('binance', 'finished')
        except Exception as e:
            set_status('binance', f'error: {e}')
            logging.error(f"[ERROR] Binance demo failed: {e}")

def run_telegram_demo():
    if ENABLE_TELEGRAM:
        set_status('telegram', 'running')
        try:
            print("\n📲 Telegram Bot Demo:\n")
            send_telegram_message("GoQuant: New signal generated! 🚦")
            set_status('telegram', 'finished')
        except Exception as e:
            set_status('telegram', f'error: {e}')
            logging.error(f"[ERROR] Telegram demo failed: {e}")

def run_slack_demo():
    if ENABLE_SLACK:
        set_status('slack', 'running')
        try:
            print("\n💬 Slack Bot Demo:\n")
            send_slack_message("GoQuant: New signal generated! 🚦")
            set_status('slack', 'finished')
        except Exception as e:
            set_status('slack', f'error: {e}')
            logging.error(f"[ERROR] Slack demo failed: {e}")

# --- Summary/Analytics Thread ---
def summary_loop():
    while True:
        scores = None
        signals = []
        highlighted = []
        portfolio_risk = {}
        try:
            scores = compute_scores()
        except Exception as e:
            print(f"[WARN] summary_loop: error in compute_scores: {e}")
            scores = {}
        try:
            signals = generate_signals(top_n=10)
            write_signals_to_csv(signals)
        except Exception as e:
            print(f"[WARN] summary_loop: error in generate_signals: {e}")
            signals = []
        try:
            highlighted = highlight_signal_changes(signals)
        except Exception as e:
            print(f'[WARN] highlight_signal_changes: {e}. Showing raw signals.')
            highlighted = [(s, True) for s in signals]
        try:
            portfolio_risk = get_portfolio_risk()
            if 'equity_curve' in portfolio_risk and portfolio_risk['equity_curve'] is not None:
                import pandas as pd
                import numpy as np
                eq = portfolio_risk['equity_curve']
                if isinstance(eq, np.ndarray):
                    portfolio_risk['equity_curve'] = pd.Series(eq)
        except Exception as e:
            print(f"[WARN] summary_loop: error in get_portfolio_risk: {e}")
            portfolio_risk = {}
        # --- Print the rich summary ---
        print("\n📊 LIVE SUMMARY")
        print(f"{'Metric':<25}{'Value'}")
        print(f"{'-'*35}")
        try:
            print(f"{'Rolling Avg':<25}{scores.get('rolling_avg', 'n/a')}")
            print(f"{'Weighted Score':<25}{scores.get('weighted_score', 'n/a')}")
            print(f"{'Time-Decayed Score':<25}{scores.get('time_decayed_score', 'n/a')}")
            print(f"{'Fear-Greed Index':<25}{scores.get('label', 'n/a')}")
        except Exception:
            print(f"{'Rolling Avg':<25}n/a")
            print(f"{'Weighted Score':<25}n/a")
            print(f"{'Time-Decayed Score':<25}n/a")
            print(f"{'Fear-Greed Index':<25}n/a")
        print(f"{'--- Top Signals ---':<25}")
        try:
            for idx, (signal, changed) in enumerate(highlighted):
                try:
                    asset = signal.get('asset', 'n/a')
                    asset_score = signal.get('asset_score', 'n/a')
                    action = signal.get('action', 'n/a')
                    confidence = signal.get('confidence', 'n/a')
                    risk = signal.get('risk', 'n/a')
                    position_size = signal.get('position_size', 'n/a')
                    holding_period = signal.get('expected_holding_period', 'n/a')
                    stop_loss = signal.get('stop_loss', 'n/a')
                    explanation = signal.get('explanation', 'n/a')
                    if changed:
                        prefix = '\033[1;32m'
                        suffix = '\033[0m'
                    else:
                        prefix = ''
                        suffix = ''
                    print(f"{prefix}#{idx+1} {action:<20} | Asset: {asset:<8} | Score: {asset_score:<6} | Conf: {confidence:<4} | Risk: {risk:<6} | Size: {position_size}% | Hold: {holding_period} | SL: {stop_loss}{suffix}")
                    print(f"   {prefix}Explanation: {explanation}{suffix}")
                except Exception as e:
                    print(f'[WARN] Error printing signal {idx}: {e}')
        except Exception as e:
            print(f"[WARN] summary_loop: error printing signals: {e}")
        print(f"{'Portfolio Volatility':<25}{portfolio_risk.get('volatility', 'n/a')}")
        print(f"{'Max Drawdown':<25}{portfolio_risk.get('max_drawdown', 'n/a')}")
        print(f"{'Value at Risk':<25}{portfolio_risk.get('value_at_risk', 'n/a')}")
        print(f"{'Risk Status':<25}{portfolio_risk.get('risk_status', 'n/a')}")
        print("-"*60)
        import time
        time.sleep(10)

# --- Main Entry Point ---
def main():
    threads = [
        threading.Thread(target=run_twitter, name="TwitterThread"),
        threading.Thread(target=run_reddit, name="RedditThread"),
        threading.Thread(target=run_news, name="NewsThread"),
        threading.Thread(target=run_finance, name="FinanceThread"),
        threading.Thread(target=run_fund_flow, name="FundFlowThread"),
        threading.Thread(target=run_economic_indicators, name="EconomicIndicatorsThread"),
        threading.Thread(target=summary_loop, name="SummaryThread"),
        threading.Thread(target=run_alpha_vantage, name="AlphaVantageThread"),
        threading.Thread(target=run_newsapi, name="NewsAPIThread"),
        threading.Thread(target=run_alternative_me, name="AlternativeMeThread"),
        threading.Thread(target=run_stocktwits, name="StockTwitsThread"),
        threading.Thread(target=run_alpaca_demo, name="AlpacaDemoThread"),
        threading.Thread(target=run_binance_demo, name="BinanceDemoThread"),
        threading.Thread(target=run_telegram_demo, name="TelegramDemoThread"),
        threading.Thread(target=run_slack_demo, name="SlackDemoThread")
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

if __name__ == "__main__":
    main()
