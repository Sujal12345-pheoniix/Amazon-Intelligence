"""
Amazon Product Intelligence Platform
Fake Review Detection Module

Combines:
1. TF-IDF + Logistic Regression ML classifier
2. Heuristic rule-based checks (capitalisation ratio, generic phrases, etc.)
3. Behavioural signals (rating extremity, verified purchase, helpful votes)
"""
import re
import logging
from typing import Optional

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

# ─── Training Data ────────────────────────────────────────────────────────────
# In production this would be trained on a labelled dataset (e.g. Ott et al. 2011
# deceptive opinion corpus or Amazon verified purchase ground truth).

_GENUINE_REVIEWS = [
    "The battery life on this headset is impressive, lasting well over 20 hours on a single charge.",
    "I've used this blender every day for 6 months and it still works like new. Very durable.",
    "Bought this for my daughter's birthday. She loves it. Good quality for the price.",
    "Returned after 3 weeks because the zipper broke. Customer service was helpful though.",
    "Sound quality is decent but the ear cushions are a bit stiff at first.",
    "Works exactly as described. Setup took 10 minutes and it connected easily to my TV.",
    "Not perfect but does the job. A bit loud when running but acceptable noise level.",
    "The colour in the photos looks slightly different from what arrived but still happy.",
    "Second purchase of this item. First one lasted 2 years before showing wear. Solid product.",
    "Instructions were confusing at first but YouTube tutorials helped. Works well now.",
    "Fast delivery and well packaged. The product itself matches the description accurately.",
    "I'm a professional chef and these knives hold an edge much longer than my old set.",
]

_FAKE_REVIEWS = [
    "BEST PRODUCT EVER!!!!! BUY NOW!! AMAZING AMAZING AMAZING YOU WON'T REGRET!!!!!",
    "Perfect product. Five stars. Highly recommend. Best purchase. Amazing quality. Buy it.",
    "I received this product for free in exchange for my honest review. It is absolutely perfect in every way!",
    "This changed my life!! Every single person needs this product!! Share with everyone you know!!",
    "Great great great great great great great great product great great quality great great great.",
    "Wow incredible must buy immediately best on market top quality premium excellence outstanding superb.",
    "I tested many similar products and this one beats them all hands down. Perfect in every single way.",
    "Just received and already love it!! Exactly what I wanted! Fast shipping! 5 stars all the way!!!",
    "Zero complaints. Perfect packaging. Perfect quality. Perfect delivery. Perfect everything.",
    "My whole family bought this and we all absolutely love it without any issues whatsoever.",
    "Unbelievably good product. I have tried everything and nothing compares to this at all.",
    "Best seller for a reason! Nobody should hesitate to buy this immediately right now today!",
]

_X_train = _GENUINE_REVIEWS + _FAKE_REVIEWS
_y_train = [0] * len(_GENUINE_REVIEWS) + [1] * len(_FAKE_REVIEWS)

fake_model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 3), max_features=8000, sublinear_tf=True)),
    ("clf", LogisticRegression(max_iter=1000, C=0.8, class_weight="balanced", random_state=42)),
])
fake_model.fit(_X_train, _y_train)
logger.info("✅ Fake review ML classifier trained and ready.")


# ─── Heuristic Rules ─────────────────────────────────────────────────────────
_GENERIC_PHRASES = [
    "best product ever", "highly recommend", "must buy", "perfect in every way",
    "five stars", "changed my life", "buy it now", "don't hesitate",
    "received for free", "in exchange for", "complimentary", "discount",
    "no complaints", "absolutely perfect", "zero complaints"
]

_SPAM_PATTERNS = [
    r"!!{3,}",            # 3+ exclamation marks
    r"\?{3,}",            # 3+ question marks
    r"(.)\1{4,}",         # 5+ repeated characters
    r"\b(\w+)(\s+\1){3,}",  # word repeated 4+ times consecutively
]


def _heuristic_check(
    text: str,
    rating: Optional[float] = None,
    verified_purchase: bool = False,
    helpful_votes: int = 0,
    total_votes: int = 0,
) -> tuple[float, list[str]]:
    """
    Returns (heuristic_fake_score 0-1, list_of_flags).
    """
    flags = []
    score = 0.0
    text_lower = text.lower()

    # 1. CAPS ratio
    alpha_chars = [c for c in text if c.isalpha()]
    if alpha_chars:
        caps_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        if caps_ratio > 0.4:
            flags.append(f"High capitalisation ratio ({caps_ratio:.0%})")
            score += 0.25

    # 2. Exclamation density
    excl_count = text.count("!")
    words = text.split()
    word_count = max(len(words), 1)
    if excl_count / word_count > 0.15:
        flags.append(f"Excessive exclamation marks ({excl_count})")
        score += 0.20

    # 3. Generic marketing phrases
    found_phrases = [p for p in _GENERIC_PHRASES if p in text_lower]
    if found_phrases:
        flags.append(f"Generic marketing phrases: {', '.join(found_phrases[:3])}")
        score += min(0.30, 0.10 * len(found_phrases))

    # 4. Spam patterns
    for pattern in _SPAM_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            flags.append("Spam text pattern detected")
            score += 0.15
            break

    # 5. Very short review with extreme rating
    if rating is not None:
        if word_count < 10 and rating == 5.0:
            flags.append("Extremely short 5-star review")
            score += 0.15
        if word_count < 5:
            flags.append("Review too short to be credible")
            score += 0.10

    # 6. Unverified purchase
    if not verified_purchase:
        score += 0.05
        flags.append("Unverified purchase")

    # 7. No helpful votes despite age (signal of isolation)
    if total_votes > 10 and helpful_votes == 0:
        flags.append("Zero helpful votes despite visibility")
        score += 0.10

    return min(score, 1.0), flags


def detect_fake(
    text: str,
    rating: Optional[float] = None,
    verified_purchase: bool = False,
    helpful_votes: int = 0,
    total_votes: int = 0,
) -> dict:
    """
    Detect whether a review is fake.

    Returns:
        {
            is_fake: bool,
            fake_probability: float,
            ml_score: float,
            heuristic_score: float,
            reasons: list[str]
        }
    """
    text = text.strip()
    if not text:
        return {
            "is_fake": False, "fake_probability": 0.0,
            "ml_score": 0.0, "heuristic_score": 0.0, "reasons": []
        }

    # ML probability
    ml_proba = float(fake_model.predict_proba([text[:512]])[0][1])

    # Heuristic score
    heuristic_score, flags = _heuristic_check(
        text, rating, verified_purchase, helpful_votes, total_votes
    )

    # Combine: 60% ML + 40% heuristics
    combined = 0.60 * ml_proba + 0.40 * heuristic_score
    is_fake = combined >= 0.50

    return {
        "is_fake": bool(is_fake),
        "fake_probability": round(combined, 4),
        "ml_score": round(ml_proba, 4),
        "heuristic_score": round(heuristic_score, 4),
        "reasons": flags,
    }


def batch_detect(reviews: list[dict]) -> list[dict]:
    """Detect fake reviews for a batch. Each dict should have 'text', optional 'rating'."""
    return [detect_fake(**{k: v for k, v in r.items() if k in
            ("text", "rating", "verified_purchase", "helpful_votes", "total_votes")})
            for r in reviews]
