import requests
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, ENABLE_TELEGRAM

def send_telegram_message(message):
    if not ENABLE_TELEGRAM:
        print("[Telegram] Integration disabled in config.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        resp = requests.post(url, data=data, timeout=10)
        if resp.status_code == 200:
            print("[Telegram] Message sent.")
        else:
            print(f"[Telegram] Error: {resp.text}")
    except Exception as e:
        print(f"[Telegram] Error sending message: {e}") 