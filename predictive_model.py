"""
Predictive Modeling for Sentiment-Based Price Prediction
Uses linear regression to predict price movement from sentiment scores.
"""
import numpy as np
from sklearn.linear_model import LinearRegression

class SentimentPricePredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.fitted = False

    def fit(self, sentiment_scores, prices):
        X = np.array(sentiment_scores).reshape(-1, 1)
        y = np.array(prices)
        self.model.fit(X, y)
        self.fitted = True

    def predict(self, sentiment_score):
        if not self.fitted:
            raise ValueError("Model not fitted yet.")
        return float(self.model.predict(np.array([[sentiment_score]]))[0])

# --- Example Usage ---
def train_predictor(sentiment_history, price_history):
    scores = [e['score'] for e in sentiment_history]
    prices = [e['price'] for e in price_history]
    predictor = SentimentPricePredictor()
    predictor.fit(scores, prices)
    return predictor 