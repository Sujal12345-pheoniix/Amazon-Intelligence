"""
Amazon Product Intelligence Platform
Review Summarization Service

Generates structured AI summaries (pros, cons, overall) from a collection
of reviews. Uses extractive summarisation when transformers are unavailable.
"""
import re
import logging
from collections import Counter
from typing import Optional

logger = logging.getLogger(__name__)

# ─── Attempt HuggingFace Summarization ───────────────────────────────────────
_sum_pipeline = None
_USE_HF_SUM = False

try:
    from transformers import pipeline as hf_pipeline
    _sum_pipeline = hf_pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        truncation=True,
    )
    _USE_HF_SUM = True
    logger.info("✅ HuggingFace DistilBART summarization pipeline loaded.")
except Exception as exc:
    logger.warning(f"⚠️  HuggingFace summarizer unavailable ({exc}). Using extractive fallback.")


# ─── Keyword Extraction Helpers ───────────────────────────────────────────────
_STOPWORDS = set([
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "it", "this", "that", "was", "are", "be", "have",
    "has", "had", "do", "did", "will", "would", "could", "should", "i",
    "my", "me", "we", "our", "you", "your", "they", "their", "its",
    "not", "no", "so", "just", "also", "very", "much", "more", "most",
    "some", "any", "all", "one", "two", "good", "great", "product",
    "item", "buy", "bought", "purchased", "get", "got", "use", "used",
])

_POSITIVE_ASPECTS = [
    "battery", "quality", "design", "price", "value", "performance",
    "comfortable", "durable", "fast", "easy", "setup", "packaging",
    "delivery", "shipping", "sound", "display", "camera", "build",
    "material", "customer service", "warranty", "software"
]
_NEGATIVE_ASPECTS = [
    "battery", "quality", "instructions", "noise", "size", "weight",
    "price", "plastic", "fragile", "connection", "wifi", "bluetooth",
    "charging", "heat", "slow", "broke", "defect", "return", "support"
]


def _extract_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def _score_sentence(sentence: str, sentiment: str) -> float:
    """Score a sentence's relevance based on keyword overlap."""
    words = set(sentence.lower().split())
    aspects = _POSITIVE_ASPECTS if sentiment == "positive" else _NEGATIVE_ASPECTS
    return sum(1 for a in aspects if a in sentence.lower())


def _extractive_summary(texts: list[str], n_sentences: int = 3) -> str:
    """Simple extractive summary by selecting highest-scored sentences."""
    all_sentences = []
    for text in texts:
        all_sentences.extend(_extract_sentences(text))

    if not all_sentences:
        return "Insufficient review data for summary."

    # Score by word frequency
    word_freq: Counter = Counter()
    for s in all_sentences:
        for w in s.lower().split():
            if w not in _STOPWORDS and len(w) > 3:
                word_freq[w] += 1

    def _sent_score(s):
        return sum(word_freq.get(w, 0) for w in s.lower().split() if w not in _STOPWORDS)

    ranked = sorted(all_sentences, key=_sent_score, reverse=True)
    selected = ranked[:n_sentences]
    return " ".join(selected)


def _extract_pros_cons(positive_reviews: list[str], negative_reviews: list[str]) -> dict:
    """Extract top mentioned positive and negative aspects."""
    pros: Counter = Counter()
    cons: Counter = Counter()

    for text in positive_reviews:
        for aspect in _POSITIVE_ASPECTS:
            if aspect in text.lower():
                pros[aspect] += 1

    for text in negative_reviews:
        for aspect in _NEGATIVE_ASPECTS:
            if aspect in text.lower():
                cons[aspect] += 1

    top_pros = [f"Good {a}" for a, _ in pros.most_common(4)]
    top_cons = [f"Issues with {a}" for a, _ in cons.most_common(4)]

    if not top_pros:
        top_pros = ["Generally positive customer experience"]
    if not top_cons:
        top_cons = ["No major issues reported"]

    return {"pros": top_pros, "cons": top_cons}


# ─── Public API ───────────────────────────────────────────────────────────────
def summarize_reviews(
    reviews: list[dict],
    product_name: Optional[str] = None,
) -> dict:
    """
    Generate a structured summary for a set of reviews.

    Each review dict: {text, sentiment_label, rating}

    Returns:
        {
            overall_summary: str,
            pros: list[str],
            cons: list[str],
            sentiment_breakdown: {positive, negative, neutral},
            avg_rating: float,
            review_count: int
        }
    """
    if not reviews:
        return {
            "overall_summary": "No reviews available.",
            "pros": [],
            "cons": [],
            "sentiment_breakdown": {"positive": 0, "negative": 0, "neutral": 0},
            "avg_rating": 0.0,
            "review_count": 0,
        }

    positive_texts = [r["text"] for r in reviews if r.get("sentiment_label") == "positive"]
    negative_texts = [r["text"] for r in reviews if r.get("sentiment_label") == "negative"]
    neutral_texts  = [r["text"] for r in reviews if r.get("sentiment_label") == "neutral"]

    all_texts = [r["text"] for r in reviews]
    ratings = [r["rating"] for r in reviews if r.get("rating") is not None]
    avg_rating = round(sum(ratings) / len(ratings), 2) if ratings else 0.0

    # Generate overall summary
    if _USE_HF_SUM and _sum_pipeline and len(all_texts) > 0:
        combined = " ".join(all_texts)[:1024]
        try:
            result = _sum_pipeline(combined, max_length=120, min_length=40, do_sample=False)
            overall_summary = result[0]["summary_text"]
        except Exception as exc:
            logger.warning(f"HF summarization failed: {exc}")
            overall_summary = _extractive_summary(all_texts)
    else:
        overall_summary = _extractive_summary(all_texts)

    # Prepend product name context
    if product_name:
        overall_summary = f"For '{product_name}': {overall_summary}"

    pros_cons = _extract_pros_cons(positive_texts, negative_texts)

    return {
        "overall_summary": overall_summary,
        "pros": pros_cons["pros"],
        "cons": pros_cons["cons"],
        "sentiment_breakdown": {
            "positive": len(positive_texts),
            "negative": len(negative_texts),
            "neutral": len(neutral_texts),
        },
        "avg_rating": avg_rating,
        "review_count": len(reviews),
    }
