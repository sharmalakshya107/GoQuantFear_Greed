import requests
from config import SLACK_WEBHOOK_URL, ENABLE_SLACK

def send_slack_message(message):
    if not ENABLE_SLACK:
        print("[Slack] Integration disabled in config.")
        return
    try:
        resp = requests.post(SLACK_WEBHOOK_URL, json={"text": message}, timeout=10)
        if resp.status_code == 200:
            print("[Slack] Message sent.")
        else:
            print(f"[Slack] Error: {resp.text}")
    except Exception as e:
        print(f"[Slack] Error sending message: {e}") 