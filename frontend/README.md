# IndicDoc AI - React frontend (experimental, not deployed)

A prototype web dashboard on top of the FastAPI backend (`backend/main.py`). It currently has one page,
which shows the baseline model's saved metrics from `GET /metrics`. The Analyze, Model Insights and
Evaluation links are placeholders.

**It is not part of the deployment.** The product is the Streamlit app (`app/streamlit_app.py`) plus the
REST API, both in the Docker image (see the main README). The Docker build skips this folder.

## Run it locally

```bash
# 1. Start the API, allowing this dev server's origin
INDICDOC_CORS_ORIGINS=http://localhost:5173 uvicorn backend.main:app --port 8000

# 2. Start the dev server
cd frontend
npm install
npm run dev          # http://localhost:5173
```

If the API runs on another address, set it in `frontend/.env.local`:

```
VITE_API_URL=http://localhost:8010
```

## Checks

```bash
npm test             # unit tests (Node's built-in test runner)
npm run lint
npm run build
```
