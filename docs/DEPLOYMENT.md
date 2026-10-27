# Deployment

The product ships as **one Docker image** containing the code, both trained models and their evaluation
results. It runs two services:

| Service | What it is | Port |
|---|---|---|
| `app` | Streamlit web app: upload a page or PDF, see the detected layout, read its text, browse results | 8501 |
| `api` | FastAPI REST API: `POST /analyze` returns boxes (and text with `ocr=true`) as JSON, `GET /metrics` returns saved results | 8000 |

## The models are not in git

These files are git-ignored (too large for a normal repository) and copied into the image from the folder it is
built in:

| File | Size | Where it comes from |
|---|---|---|
| `outputs/models/indicdlp_authors/doclayout_yolo_indicdlp.pt` | 161 MB | `python scripts/download_models.py` (IndicDLP authors, Zenodo, CC-BY 4.0) |
| `outputs/models/tessdata/*.traineddata` | 139 MB | `python scripts/download_models.py` (Tesseract tessdata_best) |
| `outputs/models/yolov8_best.pt` | 22 MB | trained in this project (`ml/training/train_yolo.py`) |
| `outputs/models/baseline_frcnn_best.pth` | 167 MB | trained in this project (`ml/training/train_baseline.py`) |

Run `python scripts/download_models.py` before building; copy the two trained files from this machine (release asset,
Git LFS or `scp`). The app offers whichever models are present, and falls back to a clearly labelled demo mode if none are.

## Requirements

- Docker with Compose v2 (`docker compose version`)
- About 4 GB of disk for the image and 2-3 GB of RAM at runtime. No GPU needed (CPU inference).

## Start

```bash
docker compose up --build -d
```

- App: http://localhost:8501
- API: http://localhost:8000 (interactive docs at http://localhost:8000/docs)

Check health:

```bash
docker compose ps                         # both services "healthy"
curl http://localhost:8501/_stcore/health # -> ok
curl http://localhost:8000/               # -> {"message":"Welcome to IndicDoc AI API"}
```

Analyse a page through the API:

```bash
curl -F file=@page.png "http://localhost:8000/analyze?model_type=yolov8&conf_threshold=0.5"
```

`model_type` is `doclayout` (default, most accurate), `yolov8` or `fasterrcnn`. The response holds `boxes` (x1, y1,
x2, y2 in pixels), `labels`, `class_names`, `scores`, the `conf_threshold` used (the model's recommended one unless
given), the `mode`, and `is_demo`. `mode=all` uses the model's high-recall threshold (lowest with at least 70% precision on validation pages,
never above the balanced one; more boxes, some wrong) instead of the balanced best-F1 one. Duplicate boxes (one element proposed under several labels) are
always merged. With `ocr=true` it also holds `document`: the detected script, the Tesseract languages, every
region with its `text` in reading order, and the page as `markdown`. `ocr_lang` is `auto` or Tesseract codes such as
`hin+eng` (`eng hin mar ben asm guj pan kan mal tam tel ori`).

Stop: `docker compose down`

## One command on a laptop

`bash start.sh` checks Docker (on macOS it installs Docker Desktop with Homebrew if needed), downloads the models
once, builds the image, starts the app and the API, waits until the app answers and opens http://localhost:8501.

## A public website (Hugging Face Spaces)

The app runs as a free Hugging Face Space (Docker SDK, CPU). `deploy/huggingface/Dockerfile` is the project's image
with two differences: the models are downloaded while the image is built, and Streamlit runs with XSRF protection and
CORS off, which file uploads need behind the Spaces proxy.

1. Create a Hugging Face account and an access token with **write** permission (huggingface.co/settings/tokens).
2. From the repository folder, after `bash start.sh` has built the image once:

   ```bash
   docker run --rm -it -v "$PWD:/work" -w /work indicdoc-ai:latest python deploy/deploy_hf_space.py --user <hf-name>
   ```

   (or `pip install huggingface_hub && python deploy/deploy_hf_space.py --user <hf-name>`). Paste the token when asked.
3. The Space builds for 5-10 minutes. The app is then at `https://<hf-name>-indicdoc-verify.hf.space`; the Verify Paper
   tab has example papers to try. `--private` keeps the Space to your account.

Only what the website needs is uploaded (app, API, `ml`, scripts, evaluation outputs, example papers). A Space is a
public service: do not upload confidential pre-print papers to it; for real papers run the image on the exam body's
own machine (`bash start.sh`), where nothing leaves it.

## Dual-reader verification

`POST /verify` reads every paper twice (`second_reader=true`, the default): reader A is the layout model with OCR,
reader B reads the PDF's text layer (born-digital files) or a second OCR pass, without the layout model. Findings carry
a `consensus` record (status, both readers' confidences); `consensus.dismissed` lists single-reader findings that the
agreed reading does not reproduce. A second OCR pass costs about as much as the first on scanned papers; born-digital
PDFs need none. `second_reader=false` gives the single-reader result.

## Configuration (environment variables of the `api` service)

| Variable | Default | Meaning |
|---|---|---|
| `INDICDOC_CORS_ORIGINS` | `http://localhost:5173` | Comma-separated websites allowed to call the API from a browser |
| `INDICDOC_MAX_UPLOAD_MB` | `20` | Larger uploads get HTTP 413 |
| `INDICDOC_MAX_MEGAPIXELS` | `40` | Images with more pixels get HTTP 413 (checked before decoding) |
| `INDICDOC_MAX_VERIFY_MB` | `100` | Upload limit of `POST /verify` (a whole paper, possibly scanned) |
| `INDICDOC_MAX_VERIFY_PAGES` | `60` | Pages per paper accepted by `POST /verify` |

Host ports: `INDICDOC_APP_PORT` (default 8501) and `INDICDOC_API_PORT` (default 8000), e.g.
`INDICDOC_API_PORT=8010 docker compose up -d` if port 8000 is already taken.

The Streamlit app limits uploads to 20 MB (`--server.maxUploadSize` in the Dockerfile).

## Before exposing it on the public internet

The image is production-shaped (non-root user, health checks, pinned dependencies, upload limits), but it
has **no login and no rate limiting**. Put both services behind a reverse proxy (nginx, Caddy, or a cloud
load balancer) that adds HTTPS and, if needed, authentication and rate limits.

## Updating the models

1. Retrain (see README "Reproduce the results"); the training scripts write
   `outputs/models/baseline_frcnn_best.pth` and `outputs/models/yolov8_best.pt`.
2. Re-run `scripts/evaluate.py` for both, so the app shows matching results.
3. `docker compose up --build -d`.

## Running without Docker

```bash
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py           # app on :8501
uvicorn backend.main:app --port 8000         # API on :8000
```
