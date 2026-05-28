# Sentiment Analysis API

A production-ready FastAPI application that classifies English text as **POSITIVE** or **NEGATIVE** using [distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) from Hugging Face Transformers.

## Features

| Feature | Detail |
|---|---|
| ML model | distilbert-base-uncased-finetuned-sst-2-english |
| Rate limiting | 10 requests / minute / IP via SlowAPI |
| Request logging | SQLite (`logs.db`) |
| Frontend | Jinja2 HTML + vanilla JS |
| Docs | Auto-generated Swagger UI at `/docs` |

---

## Project structure

```
fastapi-sentiment/
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── templates/
│   └── index.html       # Jinja2 HTML frontend
└── static/
    └── styles.css       # CSS styling
```

---

## Local development

### 1 — Clone and install dependencies

```bash
git clone <your-repo-url>
cd fastapi-sentiment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2 — Start the server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

> The first startup downloads the model (~260 MB). Subsequent starts use the Hugging Face cache.

### 3 — Try it

| URL | What you see |
|---|---|
| `http://localhost:8000/` | HTML frontend |
| `http://localhost:8000/docs` | Swagger UI |
| `http://localhost:8000/health` | Health check JSON |
| `http://localhost:8000/logs` | Request history JSON |

**Example curl:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I absolutely love this product!"}'
```

**Response:**

```json
{
  "sentiment": "POSITIVE",
  "score": 0.999877
}
```

---

## Deploy to Render

### Step 1 — Create a new Web Service

1. Go to [render.com](https://render.com) → **New** → **Web Service**
2. Connect your GitHub/GitLab repository

### Step 2 — Configure the service

| Setting | Value |
|---|---|
| **Runtime** | Python 3 |
| **Root Directory** | `fastapi-sentiment` *(if this folder is inside a larger repo)* |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
| **Instance Type** | Standard (1 CPU / 2 GB RAM minimum — the model needs ~600 MB RAM) |

### Step 3 — (Optional) Persistent disk for SQLite

By default Render's filesystem is ephemeral — logs will reset on each deploy. To persist the SQLite database:

1. In the service settings, add a **Disk** under **Storage**
2. Mount path: `/opt/render/project/src/fastapi-sentiment`
3. The `logs.db` file will survive deploys

### Step 4 — Health check

In **Settings → Health Check Path**, set:

```
/health
```

Render will ping this endpoint to verify the service is alive.

### Step 5 — Deploy

Push to your connected branch — Render auto-deploys on every push.

---

## API reference

### `POST /predict`

Analyse the sentiment of a text string.

**Request body:**

```json
{ "text": "string (1–5000 chars)" }
```

**Response:**

```json
{ "sentiment": "POSITIVE", "score": 0.9998 }
```

**Rate limit:** 10 requests per minute per IP. Exceeding returns `429 Too Many Requests`.

---

### `GET /logs?limit=50&offset=0`

Paginated list of past prediction requests stored in SQLite.

**Response:**

```json
{
  "count": 2,
  "results": [
    {
      "id": 2,
      "input_text": "Great experience!",
      "sentiment": "POSITIVE",
      "score": 0.9998,
      "timestamp": "2024-01-15T10:30:00+00:00"
    }
  ]
}
```

---

### `GET /health`

```json
{ "status": "ok", "model_loaded": true }
```

---

## Notes

- **Cold start**: the model download (~260 MB) happens once on first boot. Render's Standard plan gives enough RAM. Free-tier instances (512 MB) may OOM — use at least the Starter plan.
- **SQLite on Render**: ephemeral by default. Add a Disk for persistence (see Step 3).
- **Interactive docs**: visit `/docs` for the full Swagger UI.
