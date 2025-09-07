"""
NLP Engine: Unified interface for sentiment analysis and entity recognition.
Supports VADER, FinBERT, spaCy NER, and sarcasm detection.
"""
import re
from config import NLP_MODEL

# --- VADER ---
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader_analyzer = SentimentIntensityAnalyzer()
except ImportError:
    vader_analyzer = None

# --- FinBERT (optional, requires transformers) ---
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    finbert_model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    finbert_labels = ['negative', 'neutral', 'positive']
except Exception:
    finbert_tokenizer = None
    finbert_model = None
    finbert_labels = None

# --- spaCy NER (optional) ---
try:
    import spacy
    nlp_spacy = spacy.load("en_core_web_sm")
except Exception:
    nlp_spacy = None

# --- Sarcasm Detection (bonus, placeholder) ---
def detect_sarcasm(text):
    # Placeholder: real sarcasm detection would require a trained model
    if 'yeah right' in text.lower() or 'as if' in text.lower():
        return True
    return False

# --- Clean Text ---
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^\w\s$]", "", text)
    text = text.lower().strip()
    return text

# --- Sentiment Analysis ---
def get_sentiment(text):
    cleaned = clean_text(text)
    if NLP_MODEL == "finbert" and finbert_tokenizer and finbert_model:
        inputs = finbert_tokenizer(cleaned, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = finbert_model(**inputs)
            scores = outputs[0][0].softmax(dim=0)
            label_id = int(scores.argmax())
            label = finbert_labels[label_id]
            score = float(scores[label_id])
        return {"cleaned": cleaned, "score": score if label == 'positive' else -score if label == 'negative' else 0.0, "label": label}
    elif NLP_MODEL == "spacy" and nlp_spacy:
        doc = nlp_spacy(cleaned)
        # spaCy does not provide sentiment by default; fallback to VADER
        if vader_analyzer:
            sentiment = vader_analyzer.polarity_scores(cleaned)
            score = sentiment['compound']
            if score >= 0.05:
                label = "positive"
            elif score <= -0.05:
                label = "negative"
            else:
                label = "neutral"
            return {"cleaned": cleaned, "score": score, "label": label}
        else:
            return {"cleaned": cleaned, "score": 0.0, "label": "neutral"}
    elif NLP_MODEL == "vader" and vader_analyzer:
        sentiment = vader_analyzer.polarity_scores(cleaned)
        score = sentiment['compound']
        if score >= 0.05:
            label = "positive"
        elif score <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        return {"cleaned": cleaned, "score": score, "label": label}
    else:
        return {"cleaned": cleaned, "score": 0.0, "label": "neutral"}

# --- Entity Recognition ---
def get_entities(text):
    if nlp_spacy:
        doc = nlp_spacy(text)
        return [(ent.text, ent.label_) for ent in doc.ents]
    return []

# --- Unified NLP Pipeline ---
def analyze_text(text):
    sentiment = get_sentiment(text)
    entities = get_entities(text)
    sarcasm = detect_sarcasm(text)
    return {"sentiment": sentiment, "entities": entities, "sarcasm": sarcasm} 