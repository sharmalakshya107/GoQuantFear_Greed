import requests
from config import OPENAI_API_KEY, ENABLE_OPENAI
import random

def get_openai_sentiment(text, model="gpt-4o"):
    if not ENABLE_OPENAI:
        # Return random sentiment for demo/mock mode
        return {"label": random.choice(["bullish", "bearish", "neutral"]), "score": round(random.uniform(-1, 1), 4)}
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    prompt = f"Classify the sentiment of the following financial text as bullish, bearish, or neutral. Return a JSON with 'label' and 'score' (-1 to 1). Text: {text}"
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a financial sentiment analysis assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 50,
        "temperature": 0.0
    }
    try:
        resp = requests.post(url, headers=headers, json=data, timeout=15)
        result = resp.json()
        if "choices" in result:
            import json as pyjson
            content = result["choices"][0]["message"]["content"]
            try:
                sentiment = pyjson.loads(content)
                return sentiment
            except Exception:
                return {"label": "neutral", "score": 0.0}
        else:
            print(f"[OpenAI] Unexpected response: {result}")
            return {"label": "neutral", "score": 0.0}
    except Exception as e:
        print(f"[OpenAI] Error: {e}")
        return {"label": "neutral", "score": 0.0} 