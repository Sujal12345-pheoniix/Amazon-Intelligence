"""
Amazon Product Intelligence Platform
FastAPI Main Application
"""
import uuid
import logging
import random
import threading
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

# ─── Internal Imports ─────────────────────────────────────────────────────────
from backend.database.database import get_db, init_db
from backend.database.models import Review, Product, ModelRun
from backend.services.sentiment import analyze_sentiment, batch_analyze
from backend.services.fake_detector import detect_fake, batch_detect
from backend.services.summarizer import summarize_reviews
from backend.training.train import run_training

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ─── App Initialization ───────────────────────────────────────────────────────
app = FastAPI(
    title="Amazon Product Intelligence Platform",
    description="NLP-Powered Customer Review Analytics System",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
import os
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting Amazon Intelligence Platform...")
    init_db()
    _seed_demo_data()
    logger.info("✅ Database initialized and demo data seeded.")


# ─── Pydantic Schemas ─────────────────────────────────────────────────────────
class ReviewAnalysisRequest(BaseModel):
    text: str = Field(..., min_length=5, max_length=5000)
    rating: Optional[float] = Field(None, ge=1.0, le=5.0)
    verified_purchase: bool = False
    helpful_votes: int = 0
    total_votes: int = 0


class BatchReviewRequest(BaseModel):
    product_name: Optional[str] = None
    asin: Optional[str] = None
    reviews: List[ReviewAnalysisRequest]


class TrainingRequest(BaseModel):
    epochs: int = Field(3, ge=1, le=10)
    model_name: str = "distilbert-base-uncased"
    simulate: bool = True


class ReviewAnalysisResponse(BaseModel):
    sentiment_label: str
    sentiment_score: float
    star_class: int
    model_used: str
    is_fake: bool
    fake_probability: float
    fake_reasons: List[str]
    ml_score: float
    heuristic_score: float


# ─── Demo Data Seeding ────────────────────────────────────────────────────────
_SAMPLE_REVIEWS = [
    ("B08N5WRWNW", "Amazon Echo Dot (4th Gen)", "Electronics",
     "Great little speaker! The sound quality has improved a lot from previous versions. Alexa responds quickly.", 5.0, True),
    ("B08N5WRWNW", "Amazon Echo Dot (4th Gen)", "Electronics",
     "Decent product but the bass is lacking. Good for news and timers but not music.", 3.0, True),
    ("B08N5WRWNW", "Amazon Echo Dot (4th Gen)", "Electronics",
     "BEST PRODUCT EVER!!!! AMAZING AMAZING BUY NOW!!!!! CHANGED MY LIFE!!!!", 5.0, False),
    ("B08N5WRWNW", "Amazon Echo Dot (4th Gen)", "Electronics",
     "Stopped working after 3 months. Very disappointing for an Amazon product.", 1.0, True),
    ("B09B8YWXDF", "Kindle Paperwhite", "Electronics",
     "Reading on this is an absolute pleasure. The display is crisp and the battery lasts weeks.", 5.0, True),
    ("B09B8YWXDF", "Kindle Paperwhite", "Electronics",
     "Great for reading but charging is slow. Love the waterproof feature.", 4.0, True),
    ("B09B8YWXDF", "Kindle Paperwhite", "Electronics",
     "Perfect in every way! No complaints whatsoever. Best kindle ever made. Buy immediately!", 5.0, False),
    ("B09B8YWXDF", "Kindle Paperwhite", "Electronics",
     "Font options are good but the UI feels dated compared to tablets.", 3.0, True),
    ("C01MXQP29N", "Anker PowerCore Portable Charger", "Accessories",
     "This charges my phone 4 times on a single charge. Solid build quality and compact.", 5.0, True),
    ("C01MXQP29N", "Anker PowerCore Portable Charger", "Accessories",
     "Works well but takes forever to recharge itself. Useful for travel despite that.", 3.0, True),
    ("C01MXQP29N", "Anker PowerCore Portable Charger", "Accessories",
     "Zero defects! Perfect packaging! Perfect product! Must buy for everyone!!!", 5.0, False),
    ("C01MXQP29N", "Anker PowerCore Portable Charger", "Accessories",
     "The cable included felt cheap and the casing started peeling after a month.", 2.0, True),
]


def _seed_demo_data():
    """Populate the database with sample reviews if empty."""
    from backend.database.database import SessionLocal
    db = SessionLocal()
    try:
        if db.query(Review).count() > 0:
            return

        products: dict[str, Product] = {}
        for asin, name, category, text, rating, verified in _SAMPLE_REVIEWS:
            if asin not in products:
                p = Product(asin=asin, name=name, category=category)
                db.add(p)
                db.flush()
                products[asin] = p

            sentiment = analyze_sentiment(text, rating)
            fake = detect_fake(text, rating, verified)

            review = Review(
                product_id=products[asin].id,
                asin=asin,
                review_id=str(uuid.uuid4()),
                review_text=text,
                rating=rating,
                verified_purchase=verified,
                sentiment_label=sentiment["label"],
                sentiment_score=sentiment["score"],
                sentiment_class=sentiment["star_class"],
                is_fake=fake["is_fake"],
                fake_probability=fake["fake_probability"],
                fake_reasons=fake["reasons"],
                analyzed_at=datetime.utcnow(),
            )
            db.add(review)

        # Update product stats
        for asin, product in products.items():
            reviews = db.query(Review).filter(Review.asin == asin).all()
            if reviews:
                product.review_count = len(reviews)
                product.avg_rating = round(sum(r.rating for r in reviews if r.rating) / len(reviews), 2)
                product.fake_review_pct = round(
                    sum(1 for r in reviews if r.is_fake) / len(reviews) * 100, 1
                )
                pos = sum(1 for r in reviews if r.sentiment_label == "positive")
                product.sentiment_score = round(pos / len(reviews), 2)

        db.commit()
        logger.info(f"✅ Seeded {len(_SAMPLE_REVIEWS)} demo reviews.")
    except Exception as exc:
        db.rollback()
        logger.error(f"Seeding failed: {exc}")
    finally:
        db.close()


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_frontend():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return {"message": "Amazon Product Intelligence Platform API", "docs": "/api/docs"}


@app.get("/styles.css")
async def serve_css():
    css_path = os.path.join(FRONTEND_DIR, "styles.css")
    if os.path.exists(css_path):
        return FileResponse(css_path, media_type="text/css")
    raise HTTPException(status_code=404, detail="CSS not found")


@app.get("/app.js")
async def serve_js():
    js_path = os.path.join(FRONTEND_DIR, "app.js")
    if os.path.exists(js_path):
        return FileResponse(js_path, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="JS not found")


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat(), "version": "1.0.0"}


# ── Review Analysis ───────────────────────────────────────────────────────────
@app.post("/api/analyze", response_model=ReviewAnalysisResponse)
async def analyze_review(request: ReviewAnalysisRequest):
    """Analyze a single review for sentiment and fake detection."""
    sentiment = analyze_sentiment(request.text, request.rating)
    fake = detect_fake(
        request.text,
        request.rating,
        request.verified_purchase,
        request.helpful_votes,
        request.total_votes,
    )
    return ReviewAnalysisResponse(
        sentiment_label=sentiment["label"],
        sentiment_score=sentiment["score"],
        star_class=sentiment["star_class"],
        model_used=sentiment["model_used"],
        is_fake=fake["is_fake"],
        fake_probability=fake["fake_probability"],
        fake_reasons=fake["reasons"],
        ml_score=fake["ml_score"],
        heuristic_score=fake["heuristic_score"],
    )


# ── Batch Ingestion ───────────────────────────────────────────────────────────
@app.post("/api/ingest")
async def ingest_reviews(request: BatchReviewRequest, db: Session = Depends(get_db)):
    """Ingest and analyze a batch of reviews for a product."""
    asin = request.asin or f"DEMO-{uuid.uuid4().hex[:8].upper()}"
    product_name = request.product_name or "Unknown Product"

    # Get or create product
    product = db.query(Product).filter(Product.asin == asin).first()
    if not product:
        product = Product(asin=asin, name=product_name)
        db.add(product)
        db.flush()

    results = []
    for r in request.reviews:
        sentiment = analyze_sentiment(r.text, r.rating)
        fake = detect_fake(r.text, r.rating, r.verified_purchase, r.helpful_votes, r.total_votes)

        review = Review(
            product_id=product.id,
            asin=asin,
            review_id=str(uuid.uuid4()),
            review_text=r.text,
            rating=r.rating,
            verified_purchase=r.verified_purchase,
            helpful_votes=r.helpful_votes,
            total_votes=r.total_votes,
            sentiment_label=sentiment["label"],
            sentiment_score=sentiment["score"],
            sentiment_class=sentiment["star_class"],
            is_fake=fake["is_fake"],
            fake_probability=fake["fake_probability"],
            fake_reasons=fake["reasons"],
            analyzed_at=datetime.utcnow(),
        )
        db.add(review)
        results.append({"sentiment": sentiment, "fake": fake})

    # Update product stats
    all_reviews = db.query(Review).filter(Review.asin == asin).all()
    product.review_count = len(all_reviews)
    ratings = [rev.rating for rev in all_reviews if rev.rating]
    product.avg_rating = round(sum(ratings) / len(ratings), 2) if ratings else 0.0
    fake_count = sum(1 for rev in all_reviews if rev.is_fake)
    product.fake_review_pct = round(fake_count / len(all_reviews) * 100, 1) if all_reviews else 0.0
    pos_count = sum(1 for rev in all_reviews if rev.sentiment_label == "positive")
    product.sentiment_score = round(pos_count / len(all_reviews), 2) if all_reviews else 0.0

    db.commit()

    return {
        "message": f"Successfully ingested {len(request.reviews)} reviews",
        "product": {"asin": asin, "name": product_name},
        "results": results,
    }


# ── Products & Reviews ────────────────────────────────────────────────────────
@app.get("/api/products")
async def get_products(db: Session = Depends(get_db)):
    """Get all products with summary stats."""
    products = db.query(Product).all()
    return [
        {
            "id": p.id, "asin": p.asin, "name": p.name, "category": p.category,
            "avg_rating": p.avg_rating, "review_count": p.review_count,
            "fake_review_pct": p.fake_review_pct, "sentiment_score": p.sentiment_score,
        }
        for p in products
    ]


@app.get("/api/products/{asin}/reviews")
async def get_product_reviews(asin: str, limit: int = 20, db: Session = Depends(get_db)):
    """Get reviews for a specific product."""
    reviews = db.query(Review).filter(Review.asin == asin).limit(limit).all()
    return [
        {
            "id": r.id, "text": r.review_text, "rating": r.rating,
            "sentiment_label": r.sentiment_label, "sentiment_score": r.sentiment_score,
            "is_fake": r.is_fake, "fake_probability": r.fake_probability,
            "fake_reasons": r.fake_reasons, "verified_purchase": r.verified_purchase,
        }
        for r in reviews
    ]


@app.get("/api/products/{asin}/summary")
async def get_product_summary(asin: str, db: Session = Depends(get_db)):
    """Generate AI summary for a product's reviews."""
    reviews = db.query(Review).filter(Review.asin == asin).all()
    product = db.query(Product).filter(Product.asin == asin).first()

    if not reviews:
        raise HTTPException(status_code=404, detail="No reviews found for this product.")

    review_dicts = [
        {"text": r.review_text, "sentiment_label": r.sentiment_label, "rating": r.rating}
        for r in reviews
    ]
    summary = summarize_reviews(review_dicts, product.name if product else None)
    return summary


# ── Dashboard Metrics ─────────────────────────────────────────────────────────
@app.get("/api/metrics/overview")
async def get_overview_metrics(db: Session = Depends(get_db)):
    """Global platform metrics for the dashboard."""
    total_reviews = db.query(Review).count()
    total_products = db.query(Product).count()
    fake_count = db.query(Review).filter(Review.is_fake == True).count()
    positive = db.query(Review).filter(Review.sentiment_label == "positive").count()
    negative = db.query(Review).filter(Review.sentiment_label == "negative").count()
    neutral = db.query(Review).filter(Review.sentiment_label == "neutral").count()

    return {
        "total_reviews": total_reviews,
        "total_products": total_products,
        "fake_reviews": fake_count,
        "fake_percentage": round(fake_count / total_reviews * 100, 1) if total_reviews else 0,
        "sentiment": {
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
        },
        "sentiment_positive_pct": round(positive / total_reviews * 100, 1) if total_reviews else 0,
    }


@app.get("/api/metrics/sentiment-distribution")
async def get_sentiment_distribution(db: Session = Depends(get_db)):
    """Sentiment class distribution (1-5 stars)."""
    distribution = {i: 0 for i in range(1, 6)}
    reviews = db.query(Review.sentiment_class).all()
    for (cls,) in reviews:
        if cls and 1 <= cls <= 5:
            distribution[cls] += 1
    return distribution


# ── Training ──────────────────────────────────────────────────────────────────
_active_runs: dict[str, dict] = {}


def _training_thread(run_id: str, epochs: int, model_name: str, simulate: bool, db_url: str):
    """Background training thread."""
    from backend.database.database import SessionLocal
    db = SessionLocal()

    def _update_db(run_id, epoch_metrics, log_line):
        try:
            run = db.query(ModelRun).filter(ModelRun.run_id == run_id).first()
            if run:
                run.current_epoch = epoch_metrics["epoch"]
                run.train_loss = epoch_metrics["train_loss"]
                run.val_loss = epoch_metrics["val_loss"]
                run.accuracy = epoch_metrics["accuracy"]
                run.f1_score = epoch_metrics["f1_score"]
                history = run.metrics_history or []
                history.append(epoch_metrics)
                run.metrics_history = history
                run.logs = (run.logs or "") + f"\n{log_line}"
                db.commit()
            # Update in-memory cache
            if run_id in _active_runs:
                _active_runs[run_id]["metrics_history"] = run.metrics_history if run else []
                _active_runs[run_id]["current_epoch"] = epoch_metrics["epoch"]
                _active_runs[run_id]["logs"] = run.logs if run else ""
        except Exception as exc:
            logger.error(f"DB update failed: {exc}")
            db.rollback()

    try:
        result = run_training(
            run_id=run_id,
            epochs=epochs,
            model_name=model_name,
            simulate=simulate,
            db_update_callback=_update_db,
        )
        # Mark completed
        run = db.query(ModelRun).filter(ModelRun.run_id == run_id).first()
        if run:
            run.status = result.get("status", "completed")
            run.completed_at = datetime.utcnow()
            db.commit()
        if run_id in _active_runs:
            _active_runs[run_id]["status"] = result.get("status", "completed")
    except Exception as exc:
        logger.error(f"Training thread error: {exc}")
        run = db.query(ModelRun).filter(ModelRun.run_id == run_id).first()
        if run:
            run.status = "failed"
            run.logs = (run.logs or "") + f"\nERROR: {exc}"
            db.commit()
    finally:
        db.close()


@app.post("/api/train")
async def start_training(request: TrainingRequest, db: Session = Depends(get_db)):
    """Start a BERT fine-tuning job."""
    run_id = str(uuid.uuid4())[:8]

    model_run = ModelRun(
        run_id=run_id,
        experiment_name="amazon-sentiment-bert",
        model_name=request.model_name,
        status="running",
        epochs=request.epochs,
        current_epoch=0,
        metrics_history=[],
        logs=f"[{datetime.utcnow().isoformat()}] Training started...\nModel: {request.model_name}\nEpochs: {request.epochs}\nMode: {'SIMULATE' if request.simulate else 'REAL'}\n",
        started_at=datetime.utcnow(),
    )
    db.add(model_run)
    db.commit()

    _active_runs[run_id] = {
        "status": "running",
        "current_epoch": 0,
        "metrics_history": [],
        "logs": model_run.logs,
    }

    from backend.database.database import DATABASE_URL
    thread = threading.Thread(
        target=_training_thread,
        args=(run_id, request.epochs, request.model_name, request.simulate, DATABASE_URL),
        daemon=True,
    )
    thread.start()

    return {
        "run_id": run_id,
        "status": "running",
        "message": f"Training started with {request.epochs} epochs.",
    }


@app.get("/api/train/{run_id}/status")
async def get_training_status(run_id: str, db: Session = Depends(get_db)):
    """Poll training job status and metrics."""
    run = db.query(ModelRun).filter(ModelRun.run_id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found.")
    return {
        "run_id": run.run_id,
        "status": run.status,
        "epochs": run.epochs,
        "current_epoch": run.current_epoch,
        "train_loss": run.train_loss,
        "val_loss": run.val_loss,
        "accuracy": run.accuracy,
        "f1_score": run.f1_score,
        "metrics_history": run.metrics_history or [],
        "logs": (run.logs or "")[-3000:],  # last 3000 chars
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
    }


@app.get("/api/train/runs")
async def list_training_runs(db: Session = Depends(get_db)):
    """List all training runs."""
    runs = db.query(ModelRun).order_by(ModelRun.created_at.desc()).limit(10).all()
    return [
        {
            "run_id": r.run_id,
            "status": r.status,
            "model_name": r.model_name,
            "epochs": r.epochs,
            "accuracy": r.accuracy,
            "f1_score": r.f1_score,
            "started_at": r.started_at.isoformat() if r.started_at else None,
        }
        for r in runs
    ]
