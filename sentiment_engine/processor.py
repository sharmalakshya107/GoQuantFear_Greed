import re
from nlp.engine import get_sentiment as nlp_get_sentiment, get_entities, analyze_text

# Deprecated: clean_text, VADER, etc. are now handled in nlp/engine.py

def get_sentiment(text):
    return nlp_get_sentiment(text)

def get_entities(text):
    return get_entities(text)

def analyze_full_text(text):
    return analyze_text(text)
