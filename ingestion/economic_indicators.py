"""
Mock Economic Indicators & Market Events Ingestion
Simulates real-time economic data for integration into analytics and signals.
"""
import random
import time

def stream_economic_indicators(delay=10):
    indicators = [
        "inflation_rate", "interest_rate", "gdp_growth", "unemployment_rate", "fomc_event"
    ]
    while True:
        data = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "inflation_rate": round(random.uniform(1.0, 10.0), 2),
            "interest_rate": round(random.uniform(0.5, 7.0), 2),
            "gdp_growth": round(random.uniform(-2.0, 6.0), 2),
            "unemployment_rate": round(random.uniform(3.0, 10.0), 2),
            "fomc_event": random.choice(["none", "rate_hike", "rate_cut", "hold"])
        }
        yield data
        time.sleep(delay) 