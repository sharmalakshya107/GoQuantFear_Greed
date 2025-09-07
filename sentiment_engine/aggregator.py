from datetime import datetime, timedelta
import math
import threading
import csv
import os
import pandas as pd

# NOTE: Delete data/sentiment_history.csv before running a new demo to avoid old/corrupt data.
# Global sentiment buffer and lock
sentiment_buffer = []
buffer_lock = threading.Lock()
HISTORICAL_CSV = 'data/sentiment_history.csv'
# RETAGGED_CSV = 'data/sentiment_history_retagged.csv'

# --- Persistent Storage ---
def save_sentiment_history():
    with buffer_lock:
        try:
            abs_path = os.path.abspath(HISTORICAL_CSV)
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            file_exists = os.path.exists(abs_path)
            with open(abs_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=["timestamp", "score", "source", "influence", "tags"])
                # Only write header if file is new/empty
                if not file_exists or os.stat(abs_path).st_size == 0:
                    writer.writeheader()
                for entry in sentiment_buffer:
                    writer.writerow({
                        "timestamp": entry["timestamp"],
                        "score": entry["score"],
                        "source": entry["source"],
                        "influence": entry["influence"],
                        "tags": '|'.join(entry.get("tags", []))
                    })
            # --- Patch: Write pivoted time series for all assets ---
            # Load all history
            history = load_sentiment_history()
            if not history:
                return
            # Build long-form DataFrame
            rows = []
            for entry in history:
                ts = entry["timestamp"]
                score = entry["score"]
                for tag in entry.get("tags", []):
                    if tag in REQUIRED_ASSETS:
                        rows.append({"timestamp": ts, "asset": tag, "score": score})
            if not rows:
                return
            df = pd.DataFrame(rows)
            # Group by timestamp and asset, average score
            df = df.groupby(["timestamp", "asset"]).mean().reset_index()
            # Pivot to wide format: timestamp as index, assets as columns
            pivot = df.pivot(index="timestamp", columns="asset", values="score")
            pivot = pivot.fillna(0).sort_index()
            # Write to CSV
            pivot.to_csv("data/sentiment_timeseries.csv")
        except Exception as e:
            print(f"[ERROR] Could not write to {HISTORICAL_CSV} or sentiment_timeseries.csv: {e}")

def load_sentiment_history():
    path = HISTORICAL_CSV
    events = []
    if os.path.exists(path):
        import pandas as pd
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            tagval = row['tags']
            if pd.isnull(tagval) == True:
                tags = []
            else:
                tags = str(tagval).split('|')
            events.append({
                'timestamp': row['timestamp'],
                'score': row['score'],
                'source': row['source'],
                'influence': row['influence'],
                'tags': tags
            })
    return events

def add_to_sentiment_buffer(score, source, timestamp=None, influence=1.0, tags=None):
    if timestamp is None:
        timestamp = datetime.now()
    # Validate types
    if not isinstance(timestamp, datetime):
        try:
            timestamp = datetime.fromisoformat(str(timestamp))
        except Exception:
            timestamp = datetime.now()
    if tags is None:
        tags = ["MARKET"]
    if not isinstance(tags, list):
        tags = list(tags) if isinstance(tags, (tuple, set)) else [str(tags)]
    try:
        influence = float(influence)
    except Exception:
        influence = 1.0
    with buffer_lock:
        sentiment_buffer.append({
            "timestamp": timestamp,
            "score": score,
            "source": source,
            "influence": influence,
            "tags": tags
        })
    # Always flush after every entry, outside the lock
    save_sentiment_history()
    # print("[DEBUG] save_sentiment_history() called after adding sentiment.")

def compute_scores(buffer_window=timedelta(minutes=5), rolling_n=30, half_life_sec=1800, events=None):
    now = datetime.now()
    window_start = now - buffer_window
    # Use provided events if given, else use in-memory buffer
    if events is not None:
        relevant = [entry for entry in events if isinstance(entry["timestamp"], datetime) and entry["timestamp"] >= window_start]
    else:
        with buffer_lock:
            relevant = [entry for entry in sentiment_buffer if isinstance(entry["timestamp"], datetime) and entry["timestamp"] >= window_start]
    if not relevant:
        return {
            "rolling_avg": 0.0,
            "weighted_score": 0.0,
            "time_decayed_score": 0.0,
            "label": "NEUTRAL 😐",
            "top_asset": None,
            "top_asset_score": None
        }
    # 1. Rolling Average (last N items)
    last_n = relevant[-rolling_n:]
    rolling_avg = sum(e["score"] for e in last_n) / len(last_n)
    # 2. Weighted Score
    source_weights = {"twitter": 1.0, "reddit": 1.5, "news": 2.0}
    total_weighted_score = 0.0
    total_weight = 0.0
    for e in relevant:
        age = (now - e["timestamp"]).total_seconds()
        decay = max(0.1, (1 - age / buffer_window.total_seconds()))
        weight = source_weights.get(e["source"], 1.0) * e["influence"] * decay
        total_weighted_score += e["score"] * weight
        total_weight += weight
    weighted_score = total_weighted_score / total_weight if total_weight else 0.0
    # 3. Time-Decay (Exponential)
    def exp_decay(t): return math.exp(-t / half_life_sec)
    td_weighted_sum = sum(e["score"] * exp_decay((now - e["timestamp"]).total_seconds()) for e in relevant)
    td_total = sum(exp_decay((now - e["timestamp"]).total_seconds()) for e in relevant)
    time_decayed_score = td_weighted_sum / td_total if td_total else 0.0
    # 4. Fear-Greed Label
    scaled = (time_decayed_score + 1) * 50
    if scaled < 25:
        label = "EXTREME FEAR 😱"
    elif scaled < 45:
        label = "FEAR 😟"
    elif scaled < 55:
        label = "NEUTRAL 😐"
    elif scaled < 75:
        label = "GREED 😃"
    else:
        label = "EXTREME GREED 🚀"
    # 5. Top Mentioned Asset by Score
    tag_scores = {}
    for e in relevant:
        for tag in e.get("tags", []):
            if tag not in tag_scores:
                tag_scores[tag] = {"score": 0.0, "count": 0}
            tag_scores[tag]["score"] += e["score"]
            tag_scores[tag]["count"] += 1
    top_tag = None
    top_avg = float('-inf')
    for tag, data in tag_scores.items():
        avg = data["score"] / data["count"]
        if avg > top_avg:
            top_tag = tag
            top_avg = avg
    return {
        "rolling_avg": round(rolling_avg, 3),
        "weighted_score": round(weighted_score, 3),
        "time_decayed_score": round(time_decayed_score, 3),
        "label": label,
        "top_asset": top_tag,
        "top_asset_score": round(top_avg, 3) if top_tag else None
    }

def get_top_assets(buffer_window=timedelta(days=1), rolling_n=30, top_n=3, events=None):
    """
    Returns a list of the top N assets by average sentiment score in the current buffer window.
    """
    now = datetime.now()
    window_start = now - buffer_window
    if events is not None:
        relevant = [entry for entry in events if isinstance(entry["timestamp"], datetime) and entry["timestamp"] >= window_start]
    else:
        with buffer_lock:
            relevant = [entry for entry in sentiment_buffer if isinstance(entry["timestamp"], datetime) and entry["timestamp"] >= window_start]
    tag_scores = {}
    for e in relevant:
        for tag in e.get("tags", []):
            if tag not in tag_scores:
                tag_scores[tag] = {"score": 0.0, "count": 0}
            tag_scores[tag]["score"] += e["score"]
            tag_scores[tag]["count"] += 1
    # Compute average score for each tag
    tag_avg = [
        (tag, data["score"] / data["count"]) for tag, data in tag_scores.items() if data["count"] > 0
    ]
    # Sort by average score descending, then by tag name
    tag_avg.sort(key=lambda x: (-x[1], x[0]))
    return tag_avg[:top_n]

REQUIRED_ASSETS = {
    "BTC", "ETH", "DOGE", "SHIBA", "SOL", "XRP",
    "AAPL", "TSLA", "META", "AMZN", "GOOGL", "MSFT", "NFLX",
    "NIFTY", "GSPC", "NASDAQ", "DOWJONES", "RUT"
}

def aggregate_sentiment(events):
    # events: list of dicts with 'score' and 'tags'
    asset_scores = {asset: [] for asset in REQUIRED_ASSETS}
    for event in events:
        tags = event.get("tags", [])
        score = event.get("score", 0)
        for tag in tags:
            if tag in REQUIRED_ASSETS:
                asset_scores[tag].append(score)
                # print(f"[DEBUG][agg] Tag: {tag} | Score: {score}")
    # Compute average for each asset
    avg_scores = {asset: (sum(scores)/len(scores) if scores else 0) for asset, scores in asset_scores.items()}
    return avg_scores
