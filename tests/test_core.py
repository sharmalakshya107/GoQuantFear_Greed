import pytest
from nlp.engine import get_sentiment
from sentiment_engine.aggregator import add_to_sentiment_buffer, compute_scores
from signal_engine.generator import generate_signal
from datetime import datetime

# --- Core Sentiment Tests ---
def test_get_sentiment():
    result = get_sentiment("Bitcoin is going to the moon!")
    assert isinstance(result, dict)
    assert "score" in result
    assert "label" in result

def test_add_and_compute_scores():
    add_to_sentiment_buffer(0.8, "twitter", datetime.now(), influence=1.0, tags=["BTC"])
    scores = compute_scores()
    assert isinstance(scores, dict)
    assert "rolling_avg" in scores
    assert "weighted_score" in scores
    assert "label" in scores

def test_generate_signal():
    scores = {"time_decayed_score": 0.7, "label": "GREED 😃"}
    signal = generate_signal(scores)
    assert isinstance(signal, dict)
    assert signal["action"] in ["STRONG BUY 🚀", "BUY 🟢", "HOLD ⚖️", "SELL 🔻", "STRONG SELL 🚨"]
    assert 0.0 <= signal["confidence"] <= 1.0

# --- API Integration Smoke Tests ---
def test_alpha_vantage_import():
    from ingestion.alpha_vantage import stream_alpha_vantage_prices
    assert callable(stream_alpha_vantage_prices)

def test_newsapi_import():
    from ingestion.newsapi_stream import stream_newsapi_headlines
    assert callable(stream_newsapi_headlines)

def test_alternative_me_import():
    from ingestion.alternative_me import stream_alternative_me_fgi
    assert callable(stream_alternative_me_fgi)

def test_stocktwits_import():
    from ingestion.stocktwits_stream import stream_stocktwits
    assert callable(stream_stocktwits)

def test_alpaca_import():
    from ingestion.alpaca_stream import fetch_alpaca_account, fetch_alpaca_orders, submit_alpaca_order
    assert callable(fetch_alpaca_account)
    assert callable(fetch_alpaca_orders)
    assert callable(submit_alpaca_order)

def test_binance_import():
    from ingestion.binance_stream import fetch_binance_account, fetch_binance_trades, submit_binance_order
    assert callable(fetch_binance_account)
    assert callable(fetch_binance_trades)
    assert callable(submit_binance_order)

def test_telegram_import():
    from ingestion.telegram_bot import send_telegram_message
    assert callable(send_telegram_message)

def test_slack_import():
    from ingestion.slack_bot import send_slack_message
    assert callable(send_slack_message)

def test_openai_import():
    from nlp.openai_sentiment import get_openai_sentiment
    assert callable(get_openai_sentiment) 