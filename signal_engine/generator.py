import csv
import os
from datetime import datetime, timedelta

from sentiment_engine.aggregator import compute_scores, get_top_assets, load_sentiment_history, aggregate_sentiment, REQUIRED_ASSETS


SIGNALS_CSV = 'data/signals.csv'
signals_columns = [
    "timestamp", "action", "confidence", "risk", "position_size", "expected_holding_period", "stop_loss", "raw_score", "label", "asset", "asset_score", "explanation"
]


def generate_signals(scores=None, top_n=None):
    """
    Generate trade signals for all assets with recent sentiment, each with explanation.
    Returns a list of signal dicts.
    """
    import random
    import yfinance as yf
    import numpy as np
    # Load recent events (last 1 day)
    events = load_sentiment_history()
    now = datetime.now()
    window_start = now - timedelta(days=1)
    recent_events = []
    for e in events:
        ts = parse_dt(e["timestamp"])
        if ts is not None and ts >= window_start:
            recent_events.append(e)
    # Get per-asset sentiment scores
    asset_scores = aggregate_sentiment(recent_events)
    signals = []
    seen = set()
    for asset, score in asset_scores.items():
        if asset not in REQUIRED_ASSETS:
            continue
        # Only generate signals for assets with at least one recent mention
        if all(e.get("tags") is None or asset not in e.get("tags", []) for e in recent_events):
            continue
        # --- Fetch recent price history and compute volatility ---
        # Map asset to yfinance symbol
        symbol_map = {
            "BTC": "BTC-USD", "ETH": "ETH-USD", "DOGE": "DOGE-USD", "SHIBA": "SHIBA-USD", "SOL": "SOL-USD", "XRP": "XRP-USD",
            "AAPL": "AAPL", "TSLA": "TSLA", "META": "META", "AMZN": "AMZN", "GOOGL": "GOOGL", "MSFT": "MSFT", "NFLX": "NFLX",
            "NIFTY": "^NSEI", "GSPC": "^GSPC", "NASDAQ": "^IXIC", "DOWJONES": "^DJI", "RUT": "^RUT"
        }
        yf_symbol = symbol_map.get(asset, asset)
        try:
            ticker = yf.Ticker(yf_symbol)
            hist = ticker.history(period="14d")
            if not hist.empty and len(hist) > 2:
                returns = hist["Close"].pct_change().dropna()
                volatility = float(returns.rolling(window=5, min_periods=1).std().iloc[-1])
            else:
                volatility = None
        except Exception:
            volatility = None
        if score < -0.4:
            action = f"STRONG SELL  {asset}"
        elif score < -0.1:
            action = f"SELL  {asset}"
        elif score > 0.4:
            action = f"STRONG BUY  {asset}"
        elif score > 0.1:
            action = f"BUY  {asset}"
        else:
            action = f"HOLD  {asset}"
        now_str = now.strftime("%Y-%m-%d %H:%M:%S.%f")
        key = (now_str, asset)
        if key in seen:
            continue
        seen.add(key)
        confidence = min(1.0, abs(score))
        position_size = round(confidence * 100, 1)
        # --- Dynamic risk calculation (tuned for more medium/high risk) ---
        if volatility is not None:
            if abs(score) > 0.6 or volatility > 0.02 or position_size > 50:
                risk = 'high'
            elif abs(score) > 0.2 or volatility > 0.008 or position_size > 15:
                risk = 'medium'
            else:
                risk = 'low'
        else:
            if abs(score) > 0.6 or position_size > 50:
                risk = 'high'
            elif abs(score) > 0.2 or position_size > 15:
                risk = 'medium'
            else:
                risk = 'low'
        # --- Per-asset label logic ---
        if score < -0.1:
            label = "BEARISH"
        elif score > 0.1:
            label = "BULLISH"
        else:
            label = "NEUTRAL"
        holding_period = get_holding_period(action, confidence, score, volatility)
        # --- Dynamic stop loss based on volatility (tuned) ---
        stop_loss = None
        if volatility is not None:
            stop_loss_pct = min(max(2.5 * volatility, 0.005), 0.15)  # 2.5x vol, min 0.5%, max 15%
            stop_loss = f"{stop_loss_pct*100:.1f}%"
        else:
            if risk == 'high':
                stop_loss = '2%'
            elif risk == 'medium':
                stop_loss = '4%'
            else:
                stop_loss = '6%'
        # Signal action
        if score < -0.4:
            reason = f"Strong negative sentiment for {asset} across sources."
        elif score < -0.1:
            reason = f"Mild negative sentiment for {asset}."
        elif score > 0.4:
            reason = f"Strong positive sentiment for {asset} across sources."
        elif score > 0.1:
            reason = f"Mild positive sentiment for {asset}."
        else:
            reason = f"Neutral sentiment for {asset}."
        signals.append({
            "timestamp": now_str,
            "action": action,
            "confidence": round(confidence, 2),
            "risk": risk,
            "position_size": position_size,
            "expected_holding_period": holding_period,
            "stop_loss": stop_loss,
            "raw_score": round(score, 3),
            "label": label,
            "asset": asset,
            "asset_score": round(score, 3),
            "explanation": reason
        })
    # Strictly filter signals to only required assets
    signals = [s for s in signals if s["asset"] in REQUIRED_ASSETS]
    assert all(s["asset"] in REQUIRED_ASSETS for s in signals), "Non-required asset in signals!"
    if signals:
        write_signals_to_csv(signals)
    else:
        print("[DEBUG] No signals generated, signals.csv not written.")
    return signals

def generate_signal(scores):
    """
    Generate a single trade signal for the 'market' asset, for compatibility with tests/test_core.py.
    """
    score = scores.get("time_decayed_score", 0)
    confidence = min(1.0, abs(score))
    if abs(score) > 0.7:
        risk = 'high'
    elif abs(score) > 0.3:
        risk = 'medium'
    else:
        risk = 'low'
    position_size = round(confidence * 100, 1)
    if confidence > 0.8:
        holding_period = '5-10 days'
    elif confidence > 0.5:
        holding_period = '2-5 days'
    else:
        holding_period = '1-2 days'
    if risk == 'high':
        stop_loss = '2%'
    elif risk == 'medium':
        stop_loss = '4%'
    else:
        stop_loss = '6%'
    # Signal action
    if score < -0.4:
        action = "STRONG SELL 🚨"
        reason = "Strong negative sentiment across sources."
    elif score < -0.1:
        action = "SELL 🔻"
        reason = "Mild negative sentiment."
    elif score > 0.4:
        action = "STRONG BUY 🚀"
        reason = "Strong positive sentiment across sources."
    elif score > 0.1:
        action = "BUY 🟢"
        reason = "Mild positive sentiment."
    else:
        action = "HOLD ⚖️"
        reason = "Neutral sentiment."
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    return {
        "timestamp": now,
        "action": action,
        "confidence": round(confidence, 2),
        "risk": risk,
        "position_size": position_size,
        "expected_holding_period": holding_period,
        "stop_loss": stop_loss,
        "raw_score": round(score, 3),
        "label": scores.get("label", ""),
        "asset": "market",
        "asset_score": round(score, 3),
        "explanation": reason
    }

def write_signals_to_csv(signals):
    os.makedirs(os.path.dirname(SIGNALS_CSV), exist_ok=True)
    file_exists = os.path.isfile(SIGNALS_CSV)
    write_header = not file_exists or os.path.getsize(SIGNALS_CSV) == 0
    with open(SIGNALS_CSV, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=signals_columns)
        if write_header:
            writer.writeheader()
        for s in signals:
            writer.writerow(s)

# Helper for change highlighting
_last_signals = None

def highlight_signal_changes(new_signals):
    """
    Compares new_signals to the last signals and returns a list of (signal, changed:bool) tuples.
    """
    global _last_signals
    highlighted = []
    if _last_signals is None:
        highlighted = [(s, True) for s in new_signals]
    else:
        for idx, s in enumerate(new_signals):
            changed = False
            if idx >= len(_last_signals):
                changed = True
            else:
                prev = _last_signals[idx]
                # Consider changed if action or asset or confidence changed significantly
                if (s["action"] != prev["action"] or s["asset"] != prev["asset"] or abs(s["confidence"] - prev["confidence"]) > 0.05):
                    changed = True
            highlighted.append((s, changed))
    _last_signals = [dict(s) for s in new_signals]  # Deep copy
    return highlighted

def get_holding_period(action, confidence, score=None, volatility=None):
    action_upper = action.upper()
    # SELL/STRONG SELL: always exit now
    if "SELL" in action_upper:
        return "0 days (exit now)"
    # STRONG BUY/BUY: dynamic by confidence and volatility
    base_periods = [
        (0.7, "7-14 days"),
        (0.5, "5-10 days"),
        (0.3, "3-7 days"),
        (0.0, "1-3 days"),
    ]
    if "BUY" in action_upper:
        for thresh, period in base_periods:
            if confidence > thresh:
                # Adjust for volatility
                if volatility is not None:
                    if volatility > 0.02:  # high vol (lowered)
                        if period == "7-14 days": return "3-7 days"
                        if period == "5-10 days": return "2-5 days"
                        if period == "3-7 days": return "1-3 days"
                        if period == "1-3 days": return "1 day"
                    elif volatility < 0.006:  # low vol (lowered)
                        if period == "1-3 days": return "3-7 days"
                        if period == "3-7 days": return "5-10 days"
                        if period == "5-10 days": return "7-14 days"
                        if period == "7-14 days": return "14+ days"
                return period
    # HOLD: dynamic by score and volatility (relaxed for longer holds)
    if "HOLD" in action_upper:
        if score is not None and volatility is not None:
            if abs(score) < 0.02 and volatility > 0.02:
                return "1 day"
            elif abs(score) < 0.02 and volatility <= 0.02:
                return "2-5 days"
            elif abs(score) < 0.05 and volatility <= 0.012:
                return "7-14 days"
            elif abs(score) < 0.1 and volatility <= 0.008:
                return "14-30 days"
            else:
                return "1-2 days"
        elif score is not None:
            if abs(score) < 0.02:
                return "1 day"
            elif abs(score) < 0.05:
                return "2-5 days"
            elif abs(score) < 0.1:
                return "7-14 days"
            else:
                return "1-2 days"
        else:
            return "1-2 days"
    return "1-2 days"

def parse_dt(ts):
    if isinstance(ts, datetime):
        return ts
    try:
        return datetime.fromisoformat(str(ts))
    except Exception:
        return None

def generate_signals_timeseries():
    """
    Generate a full time series of signals for each day in sentiment_history.csv.
    Appends signals for each day to signals.csv for backtesting.
    """
    import yfinance as yf
    import numpy as np
    import pandas as pd
    events = load_sentiment_history()
    # Parse all timestamps and group by date
    for e in events:
        e['parsed_ts'] = parse_dt(e['timestamp'])
    events = [e for e in events if e['parsed_ts'] is not None]
    if not events:
        print("[ERROR] No valid events in sentiment history.")
        return
    df = pd.DataFrame(events)
    df['date'] = df['parsed_ts'].dt.date
    unique_dates = sorted(df['date'].unique())
    all_signals = []
    for d in unique_dates:
        day_events = df[df['date'] == d].to_dict('records')
        # Use the same logic as generate_signals, but for this day's events
        asset_scores = aggregate_sentiment(day_events)
        now = datetime.combine(d, datetime.min.time())
        seen = set()
        for asset, score in asset_scores.items():
            if asset not in REQUIRED_ASSETS:
                continue
            if all(e.get("tags") is None or asset not in e.get("tags", []) for e in day_events):
                continue
            # --- Fetch recent price history and compute volatility ---
            symbol_map = {
                "BTC": "BTC-USD", "ETH": "ETH-USD", "DOGE": "DOGE-USD", "SHIBA": "SHIBA-USD", "SOL": "SOL-USD", "XRP": "XRP-USD",
                "AAPL": "AAPL", "TSLA": "TSLA", "META": "META", "AMZN": "AMZN", "GOOGL": "GOOGL", "MSFT": "MSFT", "NFLX": "NFLX",
                "NIFTY": "^NSEI", "GSPC": "^GSPC", "NASDAQ": "^IXIC", "DOWJONES": "^DJI", "RUT": "^RUT"
            }
            yf_symbol = symbol_map.get(asset, asset)
            try:
                ticker = yf.Ticker(yf_symbol)
                hist = ticker.history(end=str(d + pd.Timedelta(days=1)), start=str(d - pd.Timedelta(days=14)))
                if not hist.empty and len(hist) > 2:
                    returns = hist["Close"].pct_change().dropna()
                    volatility = float(returns.rolling(window=5, min_periods=1).std().iloc[-1])
                else:
                    volatility = None
            except Exception:
                volatility = None
            if score < -0.4:
                action = f"STRONG SELL  {asset}"
            elif score < -0.1:
                action = f"SELL  {asset}"
            elif score > 0.4:
                action = f"STRONG BUY  {asset}"
            elif score > 0.1:
                action = f"BUY  {asset}"
            else:
                action = f"HOLD  {asset}"
            now_str = datetime.combine(d, datetime.min.time()).strftime("%Y-%m-%d %H:%M:%S.%f")
            key = (now_str, asset)
            if key in seen:
                continue
            seen.add(key)
            confidence = min(1.0, abs(score))
            position_size = round(confidence * 100, 1)
            # --- Dynamic risk calculation (tuned for more medium/high risk) ---
            if volatility is not None:
                if abs(score) > 0.6 or volatility > 0.02 or position_size > 50:
                    risk = 'high'
                elif abs(score) > 0.2 or volatility > 0.008 or position_size > 15:
                    risk = 'medium'
                else:
                    risk = 'low'
            else:
                if abs(score) > 0.6 or position_size > 50:
                    risk = 'high'
                elif abs(score) > 0.2 or position_size > 15:
                    risk = 'medium'
                else:
                    risk = 'low'
            if score < -0.1:
                label = "BEARISH"
            elif score > 0.1:
                label = "BULLISH"
            else:
                label = "NEUTRAL"
            holding_period = get_holding_period(action, confidence, score, volatility)
            # --- Dynamic stop loss based on volatility (tuned) ---
            stop_loss = None
            if volatility is not None:
                stop_loss_pct = min(max(2.5 * volatility, 0.005), 0.15)
                stop_loss = f"{stop_loss_pct*100:.1f}%"
            else:
                if risk == 'high':
                    stop_loss = '2%'
                elif risk == 'medium':
                    stop_loss = '4%'
                else:
                    stop_loss = '6%'
            if score < -0.4:
                reason = f"Strong negative sentiment for {asset} across sources."
            elif score < -0.1:
                reason = f"Mild negative sentiment for {asset}."
            elif score > 0.4:
                reason = f"Strong positive sentiment for {asset} across sources."
            elif score > 0.1:
                reason = f"Mild positive sentiment for {asset}."
            else:
                reason = f"Neutral sentiment for {asset}."
            all_signals.append({
                "timestamp": now_str,
                "action": action,
                "confidence": round(confidence, 2),
                "risk": risk,
                "position_size": position_size,
                "expected_holding_period": holding_period,
                "stop_loss": stop_loss,
                "raw_score": round(score, 3),
                "label": label,
                "asset": asset,
                "asset_score": round(score, 3),
                "explanation": reason
            })
    if all_signals:
        write_signals_to_csv(all_signals)
        print(f"[INFO] Wrote {len(all_signals)} signals to signals.csv for backtesting.")
    else:
        print("[WARN] No signals generated for time series.")
