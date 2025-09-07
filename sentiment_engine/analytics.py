"""
Advanced Analytics for Sentiment Data
- Trend/change-point detection
- Cross-asset contagion
- Correlation metrics
"""
import numpy as np
from datetime import datetime, timedelta
from sentiment_engine.aggregator import load_sentiment_history

# --- Trend/Change-Point Detection ---
def detect_trend(sentiment_scores, window=10):
    # Defensive: Ensure input is a list of numbers
    if not isinstance(sentiment_scores, (list, np.ndarray)) or len(sentiment_scores) < window:
        return 'stable', 0.0
    x = np.arange(window)
    y = np.array(sentiment_scores[-window:])
    slope = np.polyfit(x, y, 1)[0]
    if slope > 0.01:
        return 'uptrend', slope
    elif slope < -0.01:
        return 'downtrend', slope
    else:
        return 'stable', slope

# --- Cross-Asset Contagion ---
def contagion_index(history, tag1, tag2, window=50):
    # Defensive: Ensure input is a list of dicts
    if not isinstance(history, list) or not history:
        return 0.0
    s1 = [e['score'] for e in history if tag1 in e.get('tags', [])][-window:]
    s2 = [e['score'] for e in history if tag2 in e.get('tags', [])][-window:]
    if len(s1) < 2 or len(s2) < 2:
        return 0.0
    min_len = min(len(s1), len(s2))
    return float(np.corrcoef(s1[-min_len:], s2[-min_len:])[0, 1])

# --- Correlation Metrics ---
def sentiment_price_correlation(sentiment_history, price_history, window=50):
    # Defensive: Ensure input is a list of dicts
    if not isinstance(sentiment_history, list) or not isinstance(price_history, list):
        return 0.0
    s_scores = [e['score'] for e in sentiment_history][-window:]
    p_scores = [e['price'] for e in price_history][-window:]
    if len(s_scores) < 2 or len(p_scores) < 2:
        return 0.0
    min_len = min(len(s_scores), len(p_scores))
    return float(np.corrcoef(s_scores[-min_len:], p_scores[-min_len:])[0, 1])

# --- Behavioral Bias / Crowd Psychology (Placeholder) ---
def detect_behavioral_bias(history):
    """Placeholder for future behavioral bias/crowd psychology analytics."""
    # In a real system, would analyze sentiment reversals, herding, contrarian signals, etc.
    return {
        "bias_detected": False,
        "bias_type": None,
        "confidence": 0.0
    }

# --- Example Usage ---
def analytics_summary():
    history = load_sentiment_history()
    if not history:
        return "No historical sentiment data."
    # Trend detection for overall sentiment
    scores = [e['score'] for e in history if 'score' in e]
    trend, slope = detect_trend(scores)
    # Example contagion: BTC vs ETH
    contagion = contagion_index(history, 'BTC', 'ETH')
    return {
        'trend': trend,
        'trend_slope': slope,
        'btc_eth_contagion': contagion
    } 