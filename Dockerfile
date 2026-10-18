FROM python:3.11-slim

LABEL maintainer="IndicDoc AI" \
      description="Deep Learning-Based Multilingual Indian Document Layout Understanding"

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
# CPU-only PyTorch first: the default Linux wheels bundle ~2 GB of NVIDIA CUDA libraries this image never uses
RUN pip install --no-cache-dir torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Unprivileged user; files are copied already owned by it (a later chown would duplicate every file in a new layer)
RUN useradd --create-home --uid 1000 indicdoc

# Application code, trained models and evaluation results (raw training data is not included)
COPY --chown=indicdoc:indicdoc ml/ ml/
COPY --chown=indicdoc:indicdoc app/ app/
COPY --chown=indicdoc:indicdoc backend/ backend/
COPY --chown=indicdoc:indicdoc scripts/ scripts/
COPY --chown=indicdoc:indicdoc tests/ tests/
COPY --chown=indicdoc:indicdoc notebooks/ notebooks/
COPY --chown=indicdoc:indicdoc docs/ docs/
COPY --chown=indicdoc:indicdoc README.md .
COPY --chown=indicdoc:indicdoc .streamlit/ .streamlit/
COPY --chown=indicdoc:indicdoc data/processed/sample/ data/processed/sample/
COPY --chown=indicdoc:indicdoc outputs/ outputs/

# Libraries that write caches get a writable location
RUN chown indicdoc /app && mkdir -p /tmp/Ultralytics && chown indicdoc /tmp/Ultralytics
# No runtime pip installs or online checks from Ultralytics: the image must work offline and stay as built
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONUNBUFFERED=1 \
    YOLO_AUTOINSTALL=False \
    YOLO_OFFLINE=True
USER indicdoc

# Streamlit app (default). The API runs from the same image: see docker-compose.yml
EXPOSE 8501 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false", \
     "--server.maxUploadSize=20"]
