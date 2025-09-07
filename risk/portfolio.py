"""
Portfolio-Level Risk Monitoring (Placeholder)
Simulates portfolio risk metrics for integration into signal dashboard.
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import random
import yfinance as yf

SIGNALS_CSV = 'data/signals.csv'
SENTIMENT_CSV = 'data/sentiment_history.csv'

def simulate_portfolio_equity(signals_df, sentiment_df, window_days=7):
    # Defensive: Check for required columns and sufficient data
    if signals_df.empty or sentiment_df.empty:
        print('[WARN] simulate_portfolio_equity: Empty DataFrame(s). Returning n/a.')
        return None
    if 'timestamp' not in signals_df.columns or 'timestamp' not in sentiment_df.columns or 'asset' not in signals_df.columns:
        print('[WARN] simulate_portfolio_equity: Missing required columns. Returning n/a.')
        return None
    import numpy as np
    import pandas as pd
    # Use only recent data
    now = pd.Timestamp.now()
    signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
    recent_signals = signals_df[signals_df['timestamp'] > now - pd.Timedelta(days=window_days)]
    # Pivot signals: index=timestamp, columns=asset, values=score/confidence/action
    pivot = recent_signals.pivot_table(index='timestamp', columns='asset', values='confidence', fill_value=0)
    # Get price data for all assets
    price_data = {}
    for asset in pivot.columns:
        # Use the same price simulation/fetching logic as before
        # For now, simulate prices if not available
        n_points = len(pivot)
        price_series = np.full(n_points, 100.0)
        for i in range(1, n_points):
            price_series[i] = price_series[i-1] * (1 + np.random.normal(0, 0.02))
        price_data[asset] = price_series
    # Simulate portfolio
    equity = [100000.0]
    allocations = np.zeros(len(pivot.columns))  # start with zero allocation
    for t, (ts, row) in enumerate(pivot.iterrows()):
        # Determine allocation for each asset based on signal confidence
        # Normalize confidences to sum to 1 (if all zero, keep previous allocation)
        conf = row.values
        if np.sum(np.abs(conf)) > 0:
            alloc = np.abs(conf) / np.sum(np.abs(conf))
        else:
            alloc = allocations  # keep previous allocation
        # Compute return for each asset
        returns = np.zeros_like(alloc)
        if t > 0:
            for i, asset in enumerate(pivot.columns):
                prev_price = price_data[asset][t-1]
                curr_price = price_data[asset][t]
                returns[i] = (curr_price - prev_price) / prev_price
        # Portfolio return is weighted sum
        port_ret = np.dot(alloc, returns)
        equity.append(equity[-1] * (1 + port_ret))
        allocations = alloc  # update allocation
    equity = np.array(equity[1:])  # drop initial
    print(f'[DEBUG] Dynamic equity curve (first 10): {equity[:10]}')
    return equity

def get_portfolio_risk():
    import traceback
    warning = None
    # Try to import data source status from main
    try:
        from main import get_data_source_status
        status = get_data_source_status()
        if status.get('finance', '').startswith('simulated'):
            warning = 'Risk metrics are based on simulated price data.'
    except Exception:
        pass
    try:
        signals_df = pd.read_csv(SIGNALS_CSV)
        sentiment_df = pd.read_csv(SENTIMENT_CSV)
    except Exception as e:
        print(f'[WARN] get_portfolio_risk: Error loading CSVs: {e}. Returning n/a.')
        traceback.print_exc()
        return {
            'volatility': 'n/a',
            'max_drawdown': 'n/a',
            'value_at_risk': 'n/a',
            'risk_status': 'n/a',
            'warning': warning,
            'equity_curve': None,
        }
    try:
        equity_curve = simulate_portfolio_equity(signals_df, sentiment_df)
        if equity_curve is None or not hasattr(equity_curve, '__len__') or len(equity_curve) < 2:
            print(f'[WARN] get_portfolio_risk: equity_curve too short or None. signals: {len(signals_df) if signals_df is not None else "n/a"}, sentiment: {len(sentiment_df) if sentiment_df is not None else "n/a"}, equity: {len(equity_curve) if equity_curve is not None else "n/a"}')
            return {
                'volatility': 'n/a',
                'max_drawdown': 'n/a',
                'value_at_risk': 'n/a',
                'risk_status': 'n/a',
                'warning': warning,
                'equity_curve': equity_curve,
            }
        # Compute returns from equity curve
        equity_curve = pd.Series(equity_curve)
        returns = equity_curve.pct_change().dropna()
        if returns.empty:
            print(f'[WARN] get_portfolio_risk: returns empty. signals: {len(signals_df)}, sentiment: {len(sentiment_df)}, equity: {len(equity_curve)}')
            return {
                'volatility': 'n/a',
                'max_drawdown': 'n/a',
                'value_at_risk': 'n/a',
                'risk_status': 'n/a',
                'warning': warning,
                'equity_curve': equity_curve,
            }
        # Debug: print first 10 returns and min/max
        print('[DEBUG] Returns (first 10):', returns.values[:10])
        print('[DEBUG] Returns min/max:', returns.min(), returns.max())
        # Volatility (annualized)
        volatility = np.std(returns) * np.sqrt(252)
        # Max drawdown
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()
        # Value at Risk (VaR 95%) as positive loss percentage
        raw_var = float(np.percentile(returns, 5))
        min_ret = float(np.min(returns))
        if raw_var >= 0 and min_ret < 0:
            value_at_risk = abs(min_ret) * 100
        else:
            value_at_risk = abs(min(raw_var, 0.0)) * 100
        print(f"[DEBUG] VaR raw 5th percentile: {raw_var}, min return: {min_ret}, VaR: {value_at_risk}")
        # Risk status
        if volatility > 0.2 or max_drawdown < -0.2:
            risk_status = 'HIGH'
        elif volatility > 0.1 or max_drawdown < -0.1:
            risk_status = 'MEDIUM'
        else:
            risk_status = 'LOW'
        return {
            'volatility': f'{volatility:.3f}',
            'max_drawdown': f'{max_drawdown:.3f}',
            'value_at_risk': f'{value_at_risk:.2f}%',
            'risk_status': risk_status,
            'warning': warning,
            'equity_curve': equity_curve,
        }
    except Exception as e:
        print(f'[WARN] get_portfolio_risk: error in risk calculation: {e}')
        traceback.print_exc()
        print(f'[DEBUG] signals: {len(signals_df) if signals_df is not None else "n/a"}, sentiment: {len(sentiment_df) if sentiment_df is not None else "n/a"}, equity: {len(equity_curve) if "equity_curve" in locals() and equity_curve is not None else "n/a"}')
        return {
            'volatility': 'n/a',
            'max_drawdown': 'n/a',
            'value_at_risk': 'n/a',
            'risk_status': 'n/a',
            'warning': warning,
            'equity_curve': equity_curve if 'equity_curve' in locals() else None,
        } 