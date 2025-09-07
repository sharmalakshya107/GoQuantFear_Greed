"""
Visualization module for real-time sentiment and signal dashboards.
- Uses matplotlib for plots
- Uses rich for terminal dashboards
"""
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from sentiment_engine.aggregator import load_sentiment_history
from signal_engine.generator import generate_signal
import time

console = Console()

# --- Matplotlib Live Plot ---
def live_sentiment_plot(interval=10):
    plt.ion()
    fig, ax = plt.subplots()
    while True:
        history = load_sentiment_history()
        if not history:
            time.sleep(interval)
            continue
        times = [e['timestamp'] for e in history]
        scores = [e['score'] for e in history]
        ax.clear()
        ax.plot(times, scores, label='Sentiment Score')
        ax.set_title('Real-Time Sentiment Score')
        ax.set_xlabel('Time')
        ax.set_ylabel('Score')
        ax.legend()
        plt.draw()
        plt.pause(interval)

# --- Rich Terminal Dashboard ---
def print_signal_dashboard():
    signal = generate_signal()
    table = Table(title="Live Trade Signal")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    for k, v in signal.items():
        table.add_row(str(k), str(v))
    console.clear()
    console.print(table)

# --- Combined Live Dashboard ---
def live_dashboard(interval=10):
    while True:
        print_signal_dashboard()
        time.sleep(interval)

# --- Streamlit/Web Dashboard Placeholder ---
def launch_web_dashboard():
    """Placeholder for future Streamlit/web dashboard integration."""
    print("[INFO] Web dashboard not implemented. See README for future plans.") 