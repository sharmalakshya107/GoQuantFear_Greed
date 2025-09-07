"""
Backtesting Framework for GoQuant Fear & Greed Sentiment Engine
Evaluates signal effectiveness using historical sentiment and price data.
Outputs performance metrics and alpha generation report.
"""
import csv
from config import ASSETS
import yfinance as yf
import pandas as pd
from datetime import datetime

# --- Load Historical Price Data ---
def load_price_history(symbol, start, end):
    ticker = yf.Ticker(symbol)
    hist = ticker.history(start=start, end=end)
    return [{"timestamp": str(idx), "price": row["Close"]} for idx, row in hist.iterrows()]

# --- Load Signals ---
def load_signals(asset):
    signals = []
    try:
        with open('data/signals.csv', 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['asset'] == asset:
                    signals.append(row)
    except Exception as e:
        print(f"Error loading signals: {e}")
    return signals

# --- Simple Backtest ---
def backtest_signals(signal_func=None, asset="BTC", start=None, end=None):
    # Load signals for the selected asset
    signals = load_signals(asset)
    if not signals:
        return {"trades": 0, "total_return": 0, "alpha": 0, "period": "No data", "asset": asset}
    # Dynamically determine start and end dates from signals if not provided
    try:
        timestamps = [pd.to_datetime(s["timestamp"], errors='coerce') for s in signals]
        timestamps = [t for t in timestamps if not pd.isnull(t)]
        min_date = min(timestamps).date()
        max_date = max(timestamps).date()
        if not start:
            start = min_date.strftime("%Y-%m-%d")
        if not end:
            end = max_date.strftime("%Y-%m-%d")
        period = f"{start} to {end}"
    except Exception:
        period = "Unknown"
        if not start:
            start = "2023-01-01"
        if not end:
            end = "2023-12-31"
    # Load price history
    yf_symbol = asset
    if asset in ["BTC", "ETH", "DOGE", "SHIBA", "SOL", "XRP"]:
        yf_symbol = asset + "-USD"
    price_history = load_price_history(yf_symbol, start, end)
    if not price_history:
        return {"trades": 0, "total_return": 0, "alpha": 0, "period": period, "asset": asset}
    # Convert price history to DataFrame for easy lookup
    price_df = pd.DataFrame(price_history)
    # Robustly convert to datetime and handle errors
    try:
        price_df['timestamp'] = pd.to_datetime(price_df['timestamp'], errors='coerce')
        price_df = price_df.dropna(subset=['timestamp'])
        price_df['timestamp'] = price_df['timestamp'].dt.date
    except Exception as e:
        print(f"Error converting price_df['timestamp']: {e}")
        return {"trades": 0, "total_return": 0, "alpha": 0, "period": period, "asset": asset}
    # Align signals with price data
    trade_signals = []
    for s in signals:
        try:
            sig_time = pd.to_datetime(s['timestamp'], errors='coerce')
            if pd.isnull(sig_time):
                continue
            sig_time = sig_time.date()
            # Find the most recent available price on or before the signal date
            price_row = price_df[price_df['timestamp'] <= sig_time]
            if not price_row.empty:
                # Use the last available price before or on the signal date
                last_price_row = price_row.iloc[-1]
                trade_signals.append({
                    "timestamp": s['timestamp'],
                    "action": s['action'],
                    "price": float(last_price_row['price'])
                })
        except Exception as e:
            continue
    # Debug: print trade signals
    print(f"Trade signals for {asset}:")
    for ts in trade_signals:
        print(ts)
    # Simulate returns
    returns = []
    last_action = None
    last_price = None
    for entry in trade_signals:
        if last_action and last_action.startswith("BUY") and entry['action'].startswith("SELL"):
            if last_price is not None:
                returns.append((entry['price'] - last_price) / last_price)
            last_action = None
            last_price = None
        elif entry['action'].startswith("BUY"):
            last_action = entry['action']
            last_price = entry['price']
    # If there is an open BUY at the end, simulate a SELL at the last available price
    if last_action and last_action.startswith("BUY") and last_price is not None:
        final_price = trade_signals[-1]['price']
        if final_price != last_price:
            returns.append((final_price - last_price) / last_price)
    total_return = sum(returns)
    alpha = total_return / len(returns) if returns else 0.0
    # Use real period from signals if available
    report = {
        "trades": len(returns),
        "total_return": round(total_return * 100, 2),
        "alpha": round(alpha * 100, 2),
        "period": period,
        "asset": asset
    }
    return report 