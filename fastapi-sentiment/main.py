"""
Sentiment Analysis API
======================
FastAPI application that exposes:
  POST /predict  — run sentiment analysis on a text input
  GET  /logs     — paginated request history from SQLite
  GET  /health   — simple health-check endpoint
  GET  /         — Jinja2 HTML frontend

Rate limiting is enforced by SlowAPI.
Request logs are persisted to a local SQLite database.
"""

import sqlite3
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pydantic import BaseModel, Field
from transformers import pipeline

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded


# ─── Rate limiter ──────────────────────────────────────────────────────────────
# Uses the client IP address as the key; limits are declared per-route below.
limiter = Limiter(key_func=get_remote_address)


# ─── Database helpers ──────────────────────────────────────────────────────────
DB_PATH = "logs.db"


def init_db() -> None:
    """Create the request_logs table if it does not already exist."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS request_logs (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                input_text TEXT    NOT NULL,
                sentiment  TEXT    NOT NULL,
                score      REAL    NOT NULL,
                timestamp  TEXT    NOT NULL
            )
            """
        )
        conn.commit()


def save_log(input_text: str, sentiment: str, score: float) -> None:
    """Persist a single prediction result to the database."""
    timestamp = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO request_logs (input_text, sentiment, score, timestamp) VALUES (?, ?, ?, ?)",
            (input_text, sentiment, score, timestamp),
        )
        conn.commit()


def fetch_logs(limit: int = 50, offset: int = 0) -> list[dict]:
    """Return recent log entries as a list of dicts."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM request_logs ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
    return [dict(row) for row in rows]


# ─── ML model ──────────────────────────────────────────────────────────────────
# The model is loaded once at startup and reused across all requests.
# Using distilbert-base-uncased-finetuned-sst-2-english (fast, ~250 MB).
sentiment_pipeline = None


def load_model():
    """Load the Hugging Face sentiment-analysis pipeline into memory."""
    global sentiment_pipeline
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        # Explicitly set truncation so long texts don't raise an error
        truncation=True,
        max_length=512,
    )


# ─── Application lifespan ──────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup/shutdown tasks with the modern lifespan context manager."""
    # Startup: initialise DB + load model
    init_db()
    load_model()
    yield
    # Shutdown: nothing to clean up for this demo


# ─── FastAPI application ───────────────────────────────────────────────────────
app = FastAPI(
    title="Sentiment Analysis API",
    description="Analyse the sentiment of English text using Hugging Face Transformers.",
    version="1.0.0",
    lifespan=lifespan,
)

# Attach the rate-limiter and its error handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Serve files from the static/ folder at /static
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure Jinja2 to load templates from the templates/ folder
templates = Jinja2Templates(directory="templates")


# ─── Pydantic models ───────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to analyse")


class PredictResponse(BaseModel):
    sentiment: str = Field(..., description="POSITIVE or NEGATIVE")
    score: float   = Field(..., description="Model confidence (0–1)")


# ─── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index(request: Request):
    """Serve the HTML frontend."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health", tags=["Meta"])
async def health():
    """Lightweight health-check used by Render's health-check pings."""
    return {"status": "ok", "model_loaded": sentiment_pipeline is not None}


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
@limiter.limit("10/minute")          # Allow 10 requests per minute per IP
async def predict(request: Request, body: PredictRequest):
    """
    Run sentiment analysis on the provided text.

    - **text**: English text (1–5 000 characters)

    Returns the predicted **sentiment** label and model **score**.
    """
    if sentiment_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # Run inference — returns a list with one dict, e.g. [{"label": "POSITIVE", "score": 0.99}]
    result = sentiment_pipeline(body.text)[0]

    sentiment = result["label"]   # "POSITIVE" | "NEGATIVE"
    score     = round(result["score"], 6)

    # Persist the request to SQLite asynchronously (in-thread for simplicity)
    save_log(input_text=body.text, sentiment=sentiment, score=score)

    return PredictResponse(sentiment=sentiment, score=score)


@app.get("/logs", tags=["Meta"])
async def logs(limit: int = 50, offset: int = 0):
    """
    Return recent prediction logs from the SQLite database.

    - **limit**: number of records to return (default 50)
    - **offset**: pagination offset (default 0)
    """
    rows = fetch_logs(limit=min(limit, 200), offset=offset)
    return {"count": len(rows), "results": rows}
