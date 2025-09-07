"""
Mock Fund Flow Data Source
Simulates real-time fund flow data for assets.
"""
import random
import time
from config import ASSETS

def stream_fund_flows(delay=2):
    while True:
        flows = {}
        for symbol in ASSETS:
            # Simulate fund flow as a random value between -1M and +1M
            flows[symbol] = round(random.uniform(-1_000_000, 1_000_000), 2)
        yield {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "flows": flows
        }
        time.sleep(delay) 