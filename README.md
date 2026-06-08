# Amazon Product Intelligence Platform
### NLP-Powered Customer Review Analytics System

A production-ready end-to-end NLP platform for analyzing Amazon product reviews at scale — featuring sentiment classification, fake review detection, AI-generated summaries, and BERT fine-tuning with MLflow experiment tracking.

---

## 🏗️ Architecture

```
amazon-Intelligent/
├── backend/
│   ├── main.py                    # FastAPI application entry point
│   ├── database/
│   │   ├── database.py            # SQLAlchemy configuration
│   │   └── models.py              # ORM models (Review, Product, ModelRun)
│   ├── services/
│   │   ├── sentiment.py           # HuggingFace DistilBERT + sklearn fallback
│   │   ├── fake_detector.py       # TF-IDF + LR classifier + heuristic rules
│   │   └── summarizer.py          # Extractive + DistilBART summarization
│   └── training/
│       └── train.py               # BERT fine-tuning + MLflow tracking
├── frontend/
│   ├── index.html                 # Single-page dashboard
│   ├── styles.css                 # Dark glassmorphism design system
│   └── app.js                    # Chart.js dashboards + real-time polling
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
copy .env.example .env
# Edit .env as needed
```

### 3. Start the API Server

```bash
uvicorn backend.main:app --reload --port 8000
```

### 4. Open the Dashboard

Open your browser at **http://localhost:8000**

The API docs are available at **http://localhost:8000/api/docs**

---

## 🐳 Docker Deployment

```bash
# Start API + MLflow
docker-compose up --build

# API: http://localhost:8000
# MLflow UI: http://localhost:5000
```

---

## ✨ Features

### 1. Sentiment Analysis (`/api/analyze`)
- **Primary**: HuggingFace `distilbert-base-uncased-finetuned-sst-2-english`
- **Fallback**: TF-IDF + Logistic Regression (CPU-friendly)
- Outputs: label (positive/negative/neutral), confidence score, 1-5 star class

### 2. Fake Review Detection
- **ML Component**: TF-IDF + Logistic Regression classifier
- **Heuristic Rules**: Capitalisation ratio, exclamation density, generic phrase detection, spam patterns
- **Behavioural Signals**: Verified purchase status, helpful vote ratios
- Combined score: 60% ML + 40% Heuristics

### 3. AI Review Summaries (`/api/products/{asin}/summary`)
- Primary: HuggingFace DistilBART summarization
- Fallback: Extractive TF-IDF sentence ranking
- Structured output: overall summary, pros, cons, sentiment breakdown

### 4. BERT Fine-Tuning (`/api/train`)
- PyTorch training loop for multi-class sentiment classification (1-5 stars)
- MLflow metric logging: loss, accuracy, F1 per epoch
- **Simulate mode**: Realistic CPU-only demo (no GPU required)
- Real-time progress streaming via REST polling

### 5. Analytics Dashboard
- KPI cards: total reviews, positive sentiment %, fake count, product count
- Sentiment donut chart + star rating bar chart
- Product performance table with fake % indicator
- Per-product AI summary modal with pros/cons

---

## 🔌 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/analyze` | POST | Analyze single review |
| `/api/ingest` | POST | Batch ingest + analyze reviews |
| `/api/products` | GET | List all tracked products |
| `/api/products/{asin}/reviews` | GET | Get reviews for product |
| `/api/products/{asin}/summary` | GET | AI summary for product |
| `/api/metrics/overview` | GET | Platform-wide KPIs |
| `/api/metrics/sentiment-distribution` | GET | Star rating distribution |
| `/api/train` | POST | Start BERT fine-tuning job |
| `/api/train/{run_id}/status` | GET | Poll training progress |
| `/api/train/runs` | GET | List past training runs |

---

## 🧠 Technology Stack

| Layer | Technology |
|---|---|
| NLP Models | HuggingFace Transformers, PyTorch |
| ML Pipeline | Scikit-learn (TF-IDF, LR, Pipelines) |
| API | FastAPI, Pydantic, Uvicorn |
| Database | SQLAlchemy, SQLite / PostgreSQL |
| Experiment Tracking | MLflow |
| Frontend | HTML5, Vanilla CSS, Chart.js |
| Deployment | Docker, Docker Compose |
