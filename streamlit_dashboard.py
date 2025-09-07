import streamlit as st
from datetime import timedelta, datetime
from sentiment_engine.aggregator import load_sentiment_history, compute_scores, get_top_assets
from signal_engine.generator import generate_signals
from sentiment_engine.analytics import analytics_summary
from config import ASSETS
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import os
from risk.portfolio import get_portfolio_risk
import pytz

# --- File Paths and Columns ---
SENTIMENT_CSV = "data/sentiment_history.csv"
SIGNALS_CSV = "data/signals.csv"
sentiment_columns = ["timestamp", "score", "source", "influence", "tags"]
signals_columns = ["timestamp", "action", "confidence", "risk", "position_size", "expected_holding_period", "stop_loss", "raw_score", "label", "asset", "asset_score", "explanation"]

# --- Helper: Robust DataFrame Loader ---
def load_csv(path, columns, parse_dates=None):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, parse_dates=parse_dates)
            if df.empty:
                return pd.DataFrame(columns=columns)
            return df
        except Exception as e:
            st.warning(f"[ERROR] Could not load {path}: {e}")
            return pd.DataFrame(columns=columns)
    return pd.DataFrame(columns=columns)

def reload_data():
    df = load_csv(SENTIMENT_CSV, sentiment_columns, parse_dates=["timestamp"])
    df_signals = load_csv(SIGNALS_CSV, signals_columns, parse_dates=["timestamp"])
    df = clean_sentiment_df(df)
    df_signals = clean_signals_df(df_signals)
    return df, df_signals

# --- Helper: Clean DataFrame ---
def clean_sentiment_df(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0)
    df["influence"] = pd.to_numeric(df["influence"], errors="coerce").fillna(1)
    df["tags"] = df["tags"].fillna("").apply(lambda x: x.split("|") if isinstance(x, str) else [])
    # Convert tags to tuple for deduplication, then back to list
    df["_tags_tuple"] = df["tags"].apply(tuple)
    df = df.drop_duplicates(subset=["timestamp", "score", "source", "influence", "_tags_tuple"])
    df = df.drop(columns=["_tags_tuple"])
    return df

def clean_signals_df(df):
    df = df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for col in ["confidence", "position_size", "raw_score", "asset_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    # Deduplicate by all relevant columns, converting lists to tuples if needed
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, list)).any():
            df[f"_{col}_tuple"] = df[col].apply(tuple)
            subset_cols = [c for c in df.columns if c in signals_columns] + [f"_{col}_tuple"]
            df = df.drop_duplicates(subset=subset_cols)
            df = df.drop(columns=[f"_{col}_tuple"])
            break
    else:
        df = df.drop_duplicates(subset=[c for c in df.columns if c in signals_columns])
    return df

# --- Search Functionality ---
def filter_by_search(df, query):
    if not query or query.strip() == "":
        return df
    query = query.lower()
    mask = (
        df.apply(lambda row: any(query in str(val).lower() for val in row.values), axis=1)
    )
    return df[mask]

# --- Helper: Source Status ---
def get_source_status(history, window_minutes=10):
    now = datetime.now()  # Make now timezone-naive to match last_event
    window = timedelta(minutes=window_minutes)
    sources = [
        {"name": "Reddit", "key": "reddit"},
        {"name": "Twitter", "key": "twitter"},
        {"name": "StockTwits", "key": "stocktwits"},
        {"name": "Alternative.me", "key": "alternative_me"},
        {"name": "Alpha Vantage", "key": "alpha_vantage"},
        {"name": "NewsAPI", "key": "newsapi"},
    ]
    status = []
    for src in sources:
        events = [e for e in history if e['source'] == src['key']]
        if events:
            last_event = max(pd.to_datetime(e['timestamp']) for e in events)
            minutes_ago = (now - last_event).total_seconds() / 60
            state = 'up' if minutes_ago < window_minutes else 'down'
        else:
            last_event = None
            state = 'down'
        status.append({
            'name': src['name'],
            'state': state,
            'last_event': last_event
        })
    return status

# --- Helper: Download Button ---
def add_download_button(label, df, filename):
    csv = df.to_csv(index=False)
    st.download_button(label, data=csv, file_name=filename, mime='text/csv')

# --- Helper: Metric Card ---
def metric_card(label, value, delta=None, helptext=None):
    st.metric(label, value, delta=delta, help=helptext)

# --- Main App ---
st.set_page_config(page_title="GoQuant Sentiment Engine", layout="wide")

# --- Sidebar Navigation ---
with st.sidebar:
    st.title("GoQuant")
    st.markdown("---")
    nav = st.radio("Navigation", [
        "Home", "Signals", "Analytics", "Portfolio", "Performance", "Settings"
    ], index=0)
    st.markdown("---")
    st.markdown("<small>Powered by Lakshya Sharma</small>", unsafe_allow_html=True)
    theme = st.radio('Theme', ['light', 'dark'], index=0)
    st.session_state['theme'] = theme
    if st.button('Show Onboarding Guide'):
        st.info('''
        **Welcome to the GoQuant Sentiment Engine Dashboard!**
        - Use the sidebar to navigate between tabs.
        - Hover over any metric or chart for more info.
        - Use selectors and timeframes to explore analytics.
        - Download any table or chart for your own analysis.
        - Toggle between light and dark mode for your preferred viewing experience.
        - For help, contact support@goquant.io
        ''')

# --- Theme Styling ---
if st.session_state['theme'] == 'dark':
    st.markdown('<style>body { background-color: #18191A; color: #E4E6EB; } .stApp { background: #18191A; }</style>', unsafe_allow_html=True)

# --- Top Bar ---
col1, col2, col3 = st.columns([2, 6, 2])
with col1:
    st.write("")
with col2:
    search_query = st.text_input("Search assets, news, or signals", "")
with col3:
    st.image("https://randomuser.me/api/portraits/men/32.jpg", width=40)

# --- Data Reload on Every Run ---
df, df_signals = reload_data()

# --- Apply Search Filter ---
df_filtered = filter_by_search(df, search_query)
df_signals_filtered = filter_by_search(df_signals, search_query)

# --- Sort signals by most recent first ---
if not df_signals_filtered.empty and "timestamp" in df_signals_filtered.columns:
    df_signals_filtered = df_signals_filtered.sort_values("timestamp", ascending=False)

# --- Data Status/Warnings Section ---
# (Removed: No longer showing data source status on all pages)

# --- Risk Warning Section (shown on all tabs) ---
risk_metrics = get_portfolio_risk()
if risk_metrics.get('warning'):
    st.warning(risk_metrics['warning'])

# --- Main Tabs ---
if nav == "Home":
    st.header("Sentiment & Signal Overview")
    # Summary Metrics
    buffer_window = timedelta(days=30)
    events = df_filtered.to_dict("records")
    scores = compute_scores(buffer_window=buffer_window, events=events)
    if not scores.get("top_asset"):
        scores["top_asset"] = "MARKET"
        scores["top_asset_score"] = scores.get("time_decayed_score", 0)
    colA, colB, colC, colD = st.columns(4)
    with colA:
        metric_card("Top Sentiment Asset", scores.get("top_asset", "-"), scores.get("top_asset_score", 0), helptext="Asset with highest sentiment score.")
    with colB:
        metric_card("Fear & Greed Index", scores.get("label", "-"), scores.get("time_decayed_score", 0), helptext="Time-decayed sentiment index.")
    with colC:
        metric_card("Rolling Avg Sentiment", scores.get("rolling_avg", 0), helptext="Rolling average of sentiment scores.")
    with colD:
        metric_card("Weighted Score", scores.get("weighted_score", 0), helptext="Weighted by source and influence.")
    # Circular Gauge
    fg_score = (scores.get("time_decayed_score", 0) + 1) * 50
    fig_fg = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = fg_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Fear & Greed Index"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "#8ec5fc"},
            'steps' : [
                {'range': [0, 25], 'color': "#ff4b5c"},
                {'range': [25, 45], 'color': "#ffb26b"},
                {'range': [45, 55], 'color': "#f9f871"},
                {'range': [55, 75], 'color': "#6bcB77"},
                {'range': [75, 100], 'color': "#4bffa5"}
            ],
        }
    ))
    colA.plotly_chart(fig_fg, use_container_width=True)
    # Sentiment Chart
    st.subheader("Sentiment Score Over Time")
    if not df_filtered.empty:
        st.line_chart(df_filtered.set_index("timestamp")["score"])
    else:
        st.info("No sentiment data available.")
    add_download_button("Download Sentiment Data (CSV)", df_filtered, "sentiment_data.csv")
    # Top Trade Signals
    st.subheader("Top Trade Signals")
    if not df_signals_filtered.empty:
        st.dataframe(df_signals_filtered.head(10), use_container_width=True)
        add_download_button("Download Signals (CSV)", df_signals_filtered, "signals.csv")
    else:
        st.info("No signals available.")
    # Source Status
    st.subheader("Source Status")
    history = load_sentiment_history()
    source_status = get_source_status(history)
    cols = st.columns(len(source_status))
    for i, src in enumerate(source_status):
        with cols[i]:
            if src['state'] == 'up':
                st.success(f"{src['name']}", icon="✅")
            else:
                st.error(f"{src['name']}", icon="❌")
            if src['last_event']:
                st.caption(f"Last event: {src['last_event']}")
            else:
                st.caption("No recent data")

elif nav == "Signals":
    st.header("All Trade Signals")
    if not df_signals_filtered.empty:
        st.dataframe(df_signals_filtered, use_container_width=True)
        add_download_button("Download Signals (CSV)", df_signals_filtered, "signals.csv")
    else:
        st.info("No signals available.")

elif nav == "Analytics":
    st.header("Advanced Analytics & Correlation")
    # --- Dynamically gather all unique tags from sentiment data ---
    REQUIRED_ASSETS = {
        "BTC", "ETH", "DOGE", "SHIBA", "SOL", "XRP",
        "AAPL", "TSLA", "META", "AMZN", "GOOGL", "MSFT", "NFLX",
        "NIFTY", "GSPC", "NASDAQ", "DOWJONES", "RUT"
    }
    # When building asset_list for analytics:
    all_tags = set()
    for tags in df_filtered["tags"]:
        if isinstance(tags, list):
            all_tags.update([t for t in tags if t in REQUIRED_ASSETS])
    asset_list = sorted(all_tags)
    selected_assets = st.multiselect("Select assets/tags to compare", asset_list, default=asset_list[:2] if len(asset_list) >= 2 else asset_list)
    timeframe = st.selectbox("Timeframe", ["1d", "5d", "1mo", "3mo", "6mo", "1y"], index=2)
    import yfinance as yf
    if not df_filtered.empty and selected_assets:
        for tag in selected_assets:
            asset_sent = df_filtered[df_filtered["tags"].apply(lambda tags: tag in tags if isinstance(tags, list) else False)]
            st.subheader(f"{tag} Sentiment vs. Price")
            # Try to map tag to a yfinance symbol if possible
            symbol_map = {"BTC": "BTC-USD", "ETH": "ETH-USD", "DOGE": "DOGE-USD", "AAPL": "AAPL", "TSLA": "TSLA", "GSPC": "^GSPC", "NIFTY": "^NSEI"}
            asset_symbol = symbol_map.get(tag, tag)
            ticker = yf.Ticker(asset_symbol)
            price_hist = ticker.history(period=timeframe)
            if not asset_sent.empty and not price_hist.empty:
                price_hist = price_hist.reset_index()
                sentiment_series = asset_sent.set_index("timestamp")["score"]
                price_series = price_hist.set_index("Date")["Close"]
                # Make both indices tz-naive and sorted
                if hasattr(sentiment_series.index, 'tz_localize'):
                    sentiment_series.index = sentiment_series.index.tz_localize(None)
                if hasattr(price_series.index, 'tz_localize'):
                    price_series.index = price_series.index.tz_localize(None)
                sentiment_series = sentiment_series.sort_index()
                price_series = price_series.sort_index()
                # Align sentiment to price index
                sentiment_aligned = sentiment_series.reindex(price_series.index, method="nearest")
                if not isinstance(sentiment_aligned, pd.Series):
                    sentiment_aligned = pd.Series(sentiment_aligned, index=price_series.index)
                if hasattr(sentiment_aligned, 'fillna'):
                    sentiment_aligned = sentiment_aligned.fillna(0)
                st.line_chart({"Sentiment": sentiment_aligned, "Price": price_series})
                if (sentiment_aligned == 0).all():
                    st.warning(f"No sentiment data for {tag} in this timeframe.")
            elif asset_sent.empty:
                st.warning(f"No sentiment data for {tag}.")
            else:
                st.info(f"No price data for {asset_symbol}.")
            # --- Sentiment Momentum & Trend Detection ---
            if not asset_sent.empty:
                st.markdown(f"**{tag} Sentiment Momentum & Trend**")
                sent_ts = asset_sent.set_index("timestamp")["score"].sort_index()
                # Rolling mean and std
                roll_mean = sent_ts.rolling(window=5, min_periods=1).mean()
                roll_std = sent_ts.rolling(window=5, min_periods=1).std()
                # MACD (12/26 EMA)
                ema12 = sent_ts.ewm(span=12, adjust=False).mean()
                ema26 = sent_ts.ewm(span=26, adjust=False).mean()
                macd = ema12 - ema26
                signal = macd.ewm(span=9, adjust=False).mean()
                import plotly.graph_objs as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=sent_ts.index, y=sent_ts, mode='lines', name='Sentiment'))
                fig.add_trace(go.Scatter(x=roll_mean.index, y=roll_mean, mode='lines', name='Rolling Mean (5)'))
                fig.add_trace(go.Scatter(x=roll_std.index, y=roll_std, mode='lines', name='Rolling Std (5)'))
                # --- Change Point & Anomaly Detection ---
                # Rolling z-score for anomaly detection
                zscore = (sent_ts - roll_mean) / (roll_std + 1e-8)
                anomalies = zscore.abs() > 2.5
                # Mark anomalies
                fig.add_trace(go.Scatter(x=sent_ts.index[anomalies], y=sent_ts[anomalies], mode='markers', name='Anomaly', marker=dict(color='red', size=10, symbol='x')))
                # Simple change point detection: large delta in rolling mean
                changepoints = roll_mean.diff().abs() > (roll_std.mean() * 1.5)
                fig.add_trace(go.Scatter(x=roll_mean.index[changepoints], y=roll_mean[changepoints], mode='markers', name='Change Point', marker=dict(color='orange', size=12, symbol='star')))
                fig.update_layout(title=f"{tag} Sentiment Momentum & Anomalies", xaxis_title="Time", yaxis_title="Sentiment Score")
                st.plotly_chart(fig, use_container_width=True)
                # MACD plot
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(x=macd.index, y=macd, mode='lines', name='MACD'))
                fig_macd.add_trace(go.Scatter(x=signal.index, y=signal, mode='lines', name='Signal Line'))
                fig_macd.update_layout(title=f"{tag} Sentiment MACD", xaxis_title="Time", yaxis_title="MACD")
                st.plotly_chart(fig_macd, use_container_width=True)
    # Cross-Asset Sentiment Correlation Heatmap
    st.subheader("Cross-Asset Sentiment Correlation")
    import plotly.graph_objs as go
    import numpy as np
    pivot = {}
    if not df_filtered.empty:
        for tag in asset_list:
            asset_sent = df_filtered[df_filtered["tags"].apply(lambda tags: tag in tags if isinstance(tags, list) else False)]
            if not asset_sent.empty:
                asset_sent["timestamp"] = pd.to_datetime(asset_sent["timestamp"], errors="coerce")
                asset_sent = asset_sent.dropna(subset=["timestamp"])
                if not asset_sent.empty:
                    asset_sent = asset_sent.set_index("timestamp")
                    numeric_cols = asset_sent.select_dtypes(include='number').columns
                    if not asset_sent.index.isnull().all() and len(asset_sent.index) > 0:
                        ts = asset_sent[numeric_cols].resample("1min").mean().interpolate()["score"] if "score" in numeric_cols else pd.Series(dtype=float)
                        pivot[tag] = ts
            else:
                # If no data, fill with zeros for the same date range as others
                if pivot:
                    ref_index = next(iter(pivot.values())).index
                    pivot[tag] = pd.Series(0, index=ref_index)
    # Always include all assets, even if all zeros
    if len(pivot) > 1:
        corr_df = pd.DataFrame(pivot).fillna(0).corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr_df.values,
            x=corr_df.columns,
            y=corr_df.index,
            colorscale="RdBu",
            zmin=-1, zmax=1,
            colorbar=dict(title="Correlation")
        ))
        fig.update_layout(title="Cross-Asset Sentiment Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = go.Figure(data=go.Heatmap(z=np.zeros((2,2)), x=["Asset1","Asset2"], y=["Asset1","Asset2"], colorscale="RdBu", zmin=-1, zmax=1, colorbar=dict(title="Correlation")))
        fig.update_layout(title="Cross-Asset Sentiment Correlation Heatmap (Placeholder)")
        st.plotly_chart(fig, use_container_width=True)
        st.info("Not enough asset sentiment data for correlation heatmap. Collect more data for multiple assets to see correlations.")
    # Asset Comparison Table
    st.subheader("Asset Comparison Table")
    comp_rows = []
    for tag in selected_assets:
        asset_signals = [s for s in generate_signals(scores=compute_scores(buffer_window=timedelta(days=30), events=df_filtered.to_dict("records")), top_n=5) if s["asset"] == tag]
        if asset_signals:
            sig = asset_signals[0]
            comp_rows.append({
                "Asset/Tag": tag,
                "Sentiment Score": sig["asset_score"],
                "Action": sig["action"],
                "Confidence": sig["confidence"],
                "Risk": sig["risk"],
                "Position Size": sig["position_size"],
                "Hold": sig["expected_holding_period"],
                "Stop Loss": sig["stop_loss"]
            })
    if comp_rows:
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)
    else:
        st.info("No signal data for selected assets/tags.")

elif nav == "Portfolio":
    st.header("Portfolio & Risk Analytics")
    st.subheader("Current Portfolio Allocation (by Signal Size)")
    # Use all signals for allocation, not just search-filtered
    if not df_signals.empty:
        alloc = df_signals.groupby("asset")["position_size"].sum().to_dict()
        if alloc:
            fig = go.Figure(data=[go.Pie(labels=list(alloc.keys()), values=list(alloc.values()), hole=.4)])
            fig.update_traces(textinfo='percent+label')
            fig.update_layout(showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No allocation data available.")
    else:
        st.info("No signals available for allocation.")
    st.subheader("Live Risk Metrics")
    try:
        risk_metrics = get_portfolio_risk()
        if risk_metrics.get('warning'):
            st.warning(risk_metrics['warning'])
        metric_card("Portfolio Volatility", risk_metrics.get("volatility", "n/a"), helptext="Standard deviation of returns.")
        metric_card("Max Drawdown", risk_metrics.get("max_drawdown", "n/a"), helptext="Largest peak-to-trough drop.")
        metric_card("Value at Risk (VaR)", risk_metrics.get("value_at_risk", "n/a"), helptext="Potential loss at 95% confidence.")
        metric_card("Risk Status", risk_metrics.get("risk_status", "n/a"), helptext="Overall portfolio risk level.")
    except Exception as e:
        st.warning(f"Risk metrics unavailable: {e}")
    # --- Equity Curve Plot ---
    st.subheader("Simulated Portfolio Equity Curve")
    from risk.portfolio import simulate_portfolio_equity
    equity_curve = None
    try:
        equity_curve = simulate_portfolio_equity(df_signals_filtered, df_filtered, window_days=7)
    except Exception as e:
        st.info(f"Could not generate equity curve: {e}")
    if equity_curve is not None and len(equity_curve) > 1:
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(y=equity_curve, mode='lines', name='Equity'))
        fig_eq.update_layout(title="Simulated Portfolio Equity Curve", xaxis_title="Step", yaxis_title="Equity ($)")
        st.plotly_chart(fig_eq, use_container_width=True)
        # Download button for equity curve
        eq_df = pd.DataFrame({'Equity': equity_curve})
        st.download_button("Download Equity Curve (CSV)", eq_df.to_csv(index=False), file_name="equity_curve.csv", mime="text/csv")
    else:
        st.info("Not enough data to plot equity curve.")
    st.subheader("Actionable Recommendations")
    rec_rows = []
    for _, s in df_signals_filtered.iterrows():
        rec_rows.append({
            "Asset": s["asset"],
            "Action": s["action"],
            "Confidence": s["confidence"],
            "Rationale": s["explanation"]
        })
    if rec_rows:
        st.dataframe(pd.DataFrame(rec_rows), use_container_width=True)
    else:
        st.info("No recommendations available.")

elif nav == "Performance":
    st.header("Performance Metrics")
    st.subheader("System Performance Metrics")
    if not df_filtered.empty:
        df_sorted = df_filtered.sort_values("timestamp")
        if len(df_sorted) > 1:
            # Ensure timestamp is datetime and drop NaT
            df_sorted["timestamp"] = pd.to_datetime(df_sorted["timestamp"], errors="coerce")
            df_sorted = df_sorted.dropna(subset=["timestamp"])
            if not df_sorted.empty:
                time_span = (df_sorted["timestamp"].iloc[-1] - df_sorted["timestamp"].iloc[0]).total_seconds()
                throughput = len(df_sorted) / time_span if time_span > 0 else 0
                metric_card("Throughput (msgs/sec)", f"{throughput:.2f}", helptext="Messages processed per second.")
                latency = np.random.normal(120, 30, size=len(df_sorted))
                metric_card("Avg Latency (ms)", f"{np.mean(latency):.0f}", helptext="Simulated ingestion-to-signal latency.")
                signal_speed = np.random.normal(350, 50, size=len(df_sorted))
                metric_card("Avg Signal Speed (ms)", f"{np.mean(signal_speed):.0f}", helptext="Simulated signal generation speed.")
                st.subheader("Historical Throughput")
                if not df_sorted.set_index("timestamp").index.isnull().all() and len(df_sorted) > 0:
                    df_throughput = df_sorted.set_index("timestamp").resample("min").size()
                    st.line_chart(df_throughput.rename("Messages per Minute"))
                st.subheader("Latency Distribution (Simulated)")
                st.bar_chart(pd.Series(latency).rolling(10).mean())
            else:
                st.info("Not enough data for performance metrics.")
        else:
            st.info("Not enough data for performance metrics.")
    else:
        st.info("No sentiment history available.")

elif nav == "Settings":
    st.header("Settings & Customization")
    st.info("Settings and customization options will be shown here.")

# --- Add new tabs for advanced features ---
advanced_tabs = [
    "Transformer Sentiment", "Sarcasm Detection", "Price Prediction", "Contagion", "Behavioral Bias", "Alpha/Backtest"
]

if nav in advanced_tabs:
    st.header(f"{nav} (Advanced Feature)")
    if nav == "Transformer Sentiment":
        st.markdown("""
        **Ensemble Sentiment Analysis**
        - Combines VADER, FinBERT, and OpenAI (if enabled) for robust sentiment scoring.
        """)
        example_text = st.text_input("Enter text for ensemble sentiment analysis:", "Bitcoin is going to the moon!")
        from nlp.engine import get_sentiment
        from nlp.openai_sentiment import get_openai_sentiment
        try:
            finbert_result = get_sentiment(example_text)
        except Exception:
            finbert_result = {"label": "N/A", "score": 0.0}
        try:
            openai_result = get_openai_sentiment(example_text)
        except Exception:
            openai_result = {"label": "N/A", "score": 0.0}
        vader_result = finbert_result  # fallback if only VADER is available
        st.write("**VADER/FinBERT:**", finbert_result)
        st.write("**OpenAI:**", openai_result)
        # Simple ensemble: average score
        avg_score = (finbert_result.get("score", 0) + openai_result.get("score", 0)) / 2
        st.write(f"**Ensemble Score:** {avg_score:.3f}")
    elif nav == "Sarcasm Detection":
        st.markdown("""
        **Sarcasm/Irony Detection**
        - Uses a simple keyword-based model. For real ML, see future work.
        """)
        example_text = st.text_input("Enter text for sarcasm detection:", "Yeah right, this market is totally stable.")
        from nlp.engine import detect_sarcasm
        is_sarcastic = detect_sarcasm(example_text)
        st.write(f"**Sarcasm Detected:** {'Yes' if is_sarcastic else 'No'}")
    elif nav == "Price Prediction":
        st.markdown("""
        **Sentiment-Based Price Prediction**
        - Uses a simple linear regression on sentiment vs. price (demo).
        """)
        from sentiment_engine.aggregator import load_sentiment_history
        history = load_sentiment_history()
        import pandas as pd
        import numpy as np
        df = pd.DataFrame(history)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.sort_values('timestamp')
        # Always convert to Series before fillna
        if 'score' in df.columns:
            score_series = pd.Series(df['score'])
            score_series = pd.to_numeric(score_series, errors='coerce').fillna(0)
        else:
            score_series = pd.Series([0]*len(df))
        df['price'] = 100 + np.cumsum(score_series)
        from sklearn.linear_model import LinearRegression
        X = score_series.values.reshape(-1, 1)
        y = df['price'].values
        if len(X) > 1:
            model = LinearRegression().fit(X, y)
            pred = model.predict(X)
            st.line_chart(pd.DataFrame({'Actual': y, 'Predicted': pred}, index=df['timestamp']))
            st.write(f"R^2: {model.score(X, y):.3f}")
        else:
            st.info("Not enough data for prediction.")
    elif nav == "Contagion":
        st.markdown("""
        **Cross-Asset Sentiment Contagion**
        - Visualizes correlation between sentiment of two assets.
        """)
        from sentiment_engine.analytics import contagion_index
        asset1 = st.selectbox("Asset 1", ["BTC", "ETH", "AAPL", "TSLA"])
        asset2 = st.selectbox("Asset 2", ["BTC", "ETH", "AAPL", "TSLA"])
        history = load_sentiment_history()
        corr = contagion_index(history, asset1, asset2)
        st.write(f"Correlation between {asset1} and {asset2} sentiment: {corr:.3f}")
    elif nav == "Behavioral Bias":
        st.markdown("""
        **Behavioral Bias/Crowd Psychology Analytics**
        - Detects herding or contrarian signals (placeholder logic).
        """)
        from sentiment_engine.analytics import detect_behavioral_bias
        history = load_sentiment_history()
        bias = detect_behavioral_bias(history)
        st.write(bias)
    elif nav == "Alpha/Backtest":
        st.markdown("""
        **Alpha Generation & Backtesting Report**
        - Shows backtested results and alpha metrics (demo).
        """)
        # Placeholder: show random alpha
        import numpy as np
        alpha = np.random.uniform(-0.05, 0.15)
        st.metric("Backtested Alpha", f"{alpha:.2%}")
        st.info("For full results, see the demo notebook or backtesting module.")

# --- Real-time auto-refresh (every 30 seconds) ---
import time
if 'last_refresh' not in st.session_state:
    st.session_state['last_refresh'] = time.time()
if time.time() - st.session_state['last_refresh'] > 30:
    st.session_state['last_refresh'] = time.time()
    if hasattr(st, 'rerun'):
        st.rerun()
    else:
        st.experimental_rerun()

# --- Footer ---
st.markdown("""
---
<center><small>GoQuant Sentiment Engine &copy; 2025 | Version 1.0 | Built with ❤️ by Lakshya Sharma</small></center>
""", unsafe_allow_html=True) 