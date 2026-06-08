"""
Amazon Product Intelligence Platform
Sentiment Analysis Service

Attempts to use HuggingFace DistilBERT pipeline.
Falls back to a TF-IDF + Logistic Regression model if transformers are
unavailable or the environment has limited compute.
"""
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ─── Attempt HuggingFace pipeline ────────────────────────────────────────────
_hf_pipeline = None
_USE_HF = False

try:
    from transformers import pipeline as hf_pipeline_fn
    _hf_pipeline = hf_pipeline_fn(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True,
        max_length=512,
    )
    _USE_HF = True
    logger.info("✅ HuggingFace DistilBERT sentiment pipeline loaded.")
except Exception as exc:
    logger.warning(f"⚠️  HuggingFace pipeline unavailable ({exc}). Using fallback ML model.")

# ─── Fallback: Sklearn Sentiment Model ───────────────────────────────────────
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

_POSITIVE_WORDS = [
    "excellent", "amazing", "love", "perfect", "great", "awesome",
    "fantastic", "wonderful", "best", "highly recommend", "happy",
    "pleased", "satisfied", "quality", "durable", "impressive"
]
_NEGATIVE_WORDS = [
    "terrible", "awful", "horrible", "worst", "hate", "disappointed",
    "broken", "defective", "useless", "waste", "return", "refund",
    "poor quality", "cheap", "scam", "fake", "never buy", "regret"
]

# Build a small labeled training set for fallback model
_TRAIN_TEXTS = (
    [f"This product is {w}, I really {w} it and would highly recommend" for w in _POSITIVE_WORDS] +
    [f"This is the {w} experience, total {w} product, do not buy" for w in _NEGATIVE_WORDS]
)
_TRAIN_LABELS = [1] * len(_POSITIVE_WORDS) + [0] * len(_NEGATIVE_WORDS)

_fallback_model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
    ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
])
_fallback_model.fit(_TRAIN_TEXTS, _TRAIN_LABELS)
logger.info("✅ Fallback sklearn sentiment model ready.")


# ─── Rating to sentiment mapping ─────────────────────────────────────────────
def _rating_to_class(rating: Optional[float]) -> int:
    """Map star rating (1-5) to sentiment class (1-5)."""
    if rating is None:
        return 3
    return max(1, min(5, round(rating)))


def _score_to_star_class(pos_score: float) -> int:
    """Map a 0-1 positive probability to a 1-5 star rating class."""
    if pos_score >= 0.90:
        return 5
    elif pos_score >= 0.70:
        return 4
    elif pos_score >= 0.45:
        return 3
    elif pos_score >= 0.25:
        return 2
    else:
        return 1


# ─── Public API ──────────────────────────────────────────────────────────────
def analyze_sentiment(text: str, rating: Optional[float] = None) -> dict:
    """
    Analyze sentiment of a review text.

    Returns:
        {
            label: "positive" | "negative" | "neutral",
            score: float (0-1 confidence),
            star_class: int (1-5),
            model_used: str
        }
    """
    text = text.strip()
    if not text:
        return {"label": "neutral", "score": 0.5, "star_class": 3, "model_used": "none"}

    # Truncate very long texts
    text_trunc = text[:512]

    if _USE_HF and _hf_pipeline:
        try:
            result = _hf_pipeline(text_trunc)[0]
            label_raw = result["label"].lower()  # POSITIVE / NEGATIVE
            score = float(result["score"])

            if label_raw == "positive":
                pos_score = score
                label = "positive"
            else:
                pos_score = 1 - score
                label = "negative"

            # Adjust neutral zone
            if 0.4 <= pos_score <= 0.6:
                label = "neutral"

            star_class = _score_to_star_class(pos_score)
            if rating:
                # Blend model score with explicit rating (60/40)
                star_class = round(0.6 * star_class + 0.4 * _rating_to_class(rating))

            return {
                "label": label,
                "score": round(pos_score, 4),
                "star_class": int(star_class),
                "model_used": "distilbert-base-uncased-finetuned-sst-2-english",
            }
        except Exception as exc:
            logger.warning(f"HF inference failed: {exc}. Falling back to sklearn.")

    # Fallback sklearn model
    pos_proba = float(_fallback_model.predict_proba([text_trunc])[0][1])

    if pos_proba >= 0.60:
        label = "positive"
    elif pos_proba <= 0.40:
        label = "negative"
    else:
        label = "neutral"

    star_class = _score_to_star_class(pos_proba)
    if rating:
        star_class = round(0.6 * star_class + 0.4 * _rating_to_class(rating))

    return {
        "label": label,
        "score": round(pos_proba, 4),
        "star_class": int(star_class),
        "model_used": "tfidf-logistic-regression-fallback",
    }


def batch_analyze(reviews: list[dict]) -> list[dict]:
    """
    Analyze a batch of reviews.
    Each review dict should have: text (str), rating (optional float).
    Returns list of sentiment results merged with input.
    """
    results = []
    for r in reviews:
        result = analyze_sentiment(r.get("text", ""), r.get("rating"))
        results.append({**r, **result})
    return results
