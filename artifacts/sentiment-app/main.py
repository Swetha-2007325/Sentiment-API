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

import os
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

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
sentiment_pipeline = None


def load_model():
    """Load the Hugging Face sentiment-analysis pipeline into memory."""
    global sentiment_pipeline
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True,
        max_length=512,
    )


# ─── Application lifespan ──────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup/shutdown tasks with the modern lifespan context manager."""
    init_db()
    load_model()
    yield


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

# Serve static files from the static/ folder
app.mount("/static", StaticFiles(directory="static"), name="static")


# ─── Pydantic models ───────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to analyse")


class PredictResponse(BaseModel):
    sentiment: str = Field(..., description="POSITIVE or NEGATIVE")
    score: float   = Field(..., description="Model confidence (0–1)")


# ─── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index():
    """Serve the HTML frontend directly from disk (no Jinja2 templating needed)."""
    return FileResponse("templates/index.html")


@app.get("/health", tags=["Meta"])
async def health():
    """Lightweight health-check used by Render and Replit."""
    return {"status": "ok", "model_loaded": sentiment_pipeline is not None}


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
@limiter.limit("10/minute")
async def predict(request: Request, body: PredictRequest):
    """
    Run sentiment analysis on the provided text.
    Returns the predicted sentiment label and model confidence score.
    """
    if sentiment_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    result = sentiment_pipeline(body.text)[0]
    sentiment = result["label"]
    score = round(result["score"], 6)

    save_log(input_text=body.text, sentiment=sentiment, score=score)

    return PredictResponse(sentiment=sentiment, score=score)


@app.get("/logs", tags=["Meta"])
async def logs(limit: int = 50, offset: int = 0):
    """Return recent prediction logs from SQLite."""
    rows = fetch_logs(limit=min(limit, 200), offset=offset)
    return {"count": len(rows), "results": rows}


# ─── Entry point for local / Replit / Render ──────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    # Read PORT from environment (Replit and Render both inject this)
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
