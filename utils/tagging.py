# utils/tagging.py
import re

TAGS = {
    "BTC": r"(btc|bitcoin|#btc|\$btc)",
    "ETH": r"(eth|ethereum|#eth|\$eth)",
    "DOGE": r"(doge|dogecoin|#doge|\$doge)",
    "SHIBA": r"(shiba|shib|shiba inu|#shib|#shiba|\$shib)",
    "SOL": r"(sol|solana|#sol|\$sol)",
    "XRP": r"(xrp|ripple|#xrp|\$xrp)",
    "AAPL": r"(aapl|apple|#aapl|\$aapl)",
    "TSLA": r"(tsla|tesla|#tsla|\$tsla)",
    "MSFT": r"(msft|microsoft|#msft|\$msft)",
    "AMZN": r"(amzn|amazon|#amzn|\$amzn)",
    "GOOGL": r"(googl|google|alphabet|#googl|\$googl|#google|\$google)",
    "META": r"(meta|facebook|#meta|\$meta|#facebook|\$facebook)",
    "NFLX": r"(nflx|netflix|#nflx|\$nflx)",
    "NIFTY": r"(nifty|nifty 50|#nifty|#nifty50)",
    "GSPC": r"(s\&p|s&p|s&p 500|gspc|#gspc|#spx|spx|s p 500)",
    "NASDAQ": r"(nasdaq|ixic|#ixic|#nasdaq)",
    "DOWJONES": r"(dow|dow jones|dji|#dji|#dowjones|#dow)",
    "RUT": r"(russell|russell 2000|rut|#rut)"
}

REQUIRED_ASSETS = set(TAGS.keys())

def detect_tags(text):
    tags_found = []
    cleaned = text.lower()
    # Regex or substring match
    for tag, pattern in TAGS.items():
        try:
            if re.search(pattern, cleaned, re.IGNORECASE) or tag.lower() in cleaned or tag.lower().replace('-', '') in cleaned:
                tags_found.append(tag)
        except re.error as e:
            print(f"[REGEX ERROR] Pattern issue in tag '{tag}': {e}")
    # Fallback: substring match for asset names
    if not tags_found:
        for tag in TAGS:
            if tag.lower() in cleaned or tag.lower().replace('-', '') in cleaned:
                tags_found.append(tag)
    # If still no tag, tag as MARKET
    if not tags_found:
        tags_found = ["MARKET"]
    return tags_found
