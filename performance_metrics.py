"""
Performance Metrics Module
Logs and reports throughput, latency, and signal generation speed.
"""
import time
from functools import wraps

metrics = {
    'ingest_count': 0,
    'ingest_start': None,
    'ingest_end': None,
    'signal_times': []
}

def track_ingestion(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if metrics['ingest_start'] is None:
            metrics['ingest_start'] = time.time()
        metrics['ingest_count'] += 1
        result = func(*args, **kwargs)
        metrics['ingest_end'] = time.time()
        return result
    return wrapper

def track_signal(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        metrics['signal_times'].append(end - start)
        return result
    return wrapper

def report_metrics():
    duration = (metrics['ingest_end'] - metrics['ingest_start']) if metrics['ingest_start'] and metrics['ingest_end'] else 0
    throughput = metrics['ingest_count'] / duration if duration > 0 else 0
    avg_signal_time = sum(metrics['signal_times']) / len(metrics['signal_times']) if metrics['signal_times'] else 0
    return {
        'ingest_count': metrics['ingest_count'],
        'duration_sec': round(duration, 2),
        'throughput_per_sec': round(throughput, 2),
        'avg_signal_latency_sec': round(avg_signal_time, 4)
    } 