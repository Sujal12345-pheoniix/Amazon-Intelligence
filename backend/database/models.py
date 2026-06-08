"""
Amazon Product Intelligence Platform
SQLAlchemy ORM Models
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship
from backend.database.database import Base


class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    asin = Column(String(20), unique=True, index=True, nullable=False)
    name = Column(String(500), nullable=False)
    category = Column(String(200))
    avg_rating = Column(Float, default=0.0)
    review_count = Column(Integer, default=0)
    fake_review_pct = Column(Float, default=0.0)
    sentiment_score = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    reviews = relationship("Review", back_populates="product")

    def __repr__(self):
        return f"<Product(asin={self.asin}, name={self.name[:30]})>"


class Review(Base):
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=True)
    asin = Column(String(20), index=True)
    review_id = Column(String(100), unique=True, index=True)
    reviewer_id = Column(String(100))
    reviewer_name = Column(String(200))
    review_text = Column(Text, nullable=False)
    summary = Column(Text)
    rating = Column(Float)
    helpful_votes = Column(Integer, default=0)
    total_votes = Column(Integer, default=0)
    verified_purchase = Column(Boolean, default=False)

    # ML Analysis Results
    sentiment_label = Column(String(20))      # positive / negative / neutral
    sentiment_score = Column(Float)           # confidence 0-1
    sentiment_class = Column(Integer)         # 1-5 star class
    is_fake = Column(Boolean, default=False)
    fake_probability = Column(Float, default=0.0)
    fake_reasons = Column(JSON)               # list of heuristic flags

    review_date = Column(DateTime)
    analyzed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

    product = relationship("Product", back_populates="reviews")

    def __repr__(self):
        return f"<Review(id={self.id}, sentiment={self.sentiment_label}, fake={self.is_fake})>"


class ModelRun(Base):
    __tablename__ = "model_runs"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String(100), unique=True, index=True)
    experiment_name = Column(String(200))
    model_name = Column(String(200))
    status = Column(String(50), default="pending")  # pending / running / completed / failed
    epochs = Column(Integer, default=3)
    current_epoch = Column(Integer, default=0)
    train_loss = Column(Float)
    val_loss = Column(Float)
    accuracy = Column(Float)
    f1_score = Column(Float)
    metrics_history = Column(JSON)  # list of {epoch, loss, accuracy, f1}
    logs = Column(Text, default="")
    mlflow_run_id = Column(String(200))
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<ModelRun(run_id={self.run_id}, status={self.status})>"
