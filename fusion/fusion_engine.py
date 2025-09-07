from collections import defaultdict
from datetime import datetime

def fuse_stream_data(data_list):
    fused = defaultdict(lambda: {"count": 0, "sentiment_sum": 0.0})

    for data in data_list:
        for tag in data.get("tags", []):
            fused[tag]["count"] += 1
            fused[tag]["sentiment_sum"] += data["sentiment"]["score"]

    result = []
    for tag, values in fused.items():
        avg_sentiment = values["sentiment_sum"] / values["count"]
        result.append({
            "tag": tag,
            "mentions": values["count"],
            "avg_sentiment": round(avg_sentiment, 4),
        })

    return sorted(result, key=lambda x: -x["mentions"])
