import yfinance as yf
from datetime import datetime
from config import ASSETS
import logging
import requests

logger = logging.getLogger("finance_data")

# Expand this mapping as you add more crypto assets to ASSETS
COINGECKO_MAP = {
    "BTC-USD": "bitcoin",
    "ETH-USD": "ethereum",
    "DOGE-USD": "dogecoin",
    "SHIBA-USD": "shiba-inu",
    "SOL-USD": "solana",
    "XRP-USD": "ripple",
    # Add more as needed
}
# Keep this mapping updated for any new crypto assets you add to ASSETS.

def fetch_coingecko_price(symbol):
    cg_id = COINGECKO_MAP.get(symbol)
    if not cg_id:
        return None
    try:
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={cg_id}&vs_currencies=usd"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return data[cg_id]["usd"]
    except Exception as e:
        logger.error(f"[CoinGecko] Error fetching {symbol}: {e}")
        return None

def fetch_price_data():
    print(f"\n📈 Market Prices @ {datetime.now()}")
    from random import gauss
    missing_prices = []
    for symbol, name in ASSETS.items():
        price = None
        used_simulated = False
        try:
            ticker = yf.Ticker(symbol)
            price = ticker.info.get("regularMarketPrice", None)
            if price:
                print(f"{name} ({symbol}): ${price}")
            else:
                # Try CoinGecko for crypto
                price = fetch_coingecko_price(symbol)
                if price:
                    print(f"{name} ({symbol}): ${price} (CoinGecko)")
                else:
                    # Simulate price if not available
                    price = 100 + gauss(0, 2)
                    used_simulated = True
                    print(f"{name} ({symbol}): ⚠️ Simulated price: ${price:.2f}")
                    try:
                        from main import set_status
                        set_status('finance', f'simulated for {symbol}')
                    except Exception:
                        pass
        except Exception as e:
            # Try CoinGecko for crypto
            price = fetch_coingecko_price(symbol)
            if price:
                print(f"{name} ({symbol}): ${price} (CoinGecko)")
            else:
                # Simulate price if API fails
                price = 100 + gauss(0, 2)
                used_simulated = True
                print(f"{name} ({symbol}): ⚠️ Simulated price (API error): ${price:.2f}")
                logger.error(f"Error fetching price for {symbol}: {e} (Simulated price used)")
                try:
                    from main import set_status
                    set_status('finance', f'simulated for {symbol}')
                except Exception:
                    pass
        if used_simulated:
            missing_prices.append((symbol, name))
    if missing_prices:
        print("\n⚠️ The following assets are missing real market prices:")
        for symbol, name in missing_prices:
            print(f"- {name} ({symbol})")
        print("\nTo improve coverage, check ticker symbols and expand COINGECKO_MAP for cryptos.")
