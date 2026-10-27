# IndicDoc Verify on Hugging Face Spaces (Docker SDK). Same image as the project's Dockerfile, except that the
# layout model and the OCR language models are downloaded while the image is built (Spaces build with internet),
# and Streamlit runs behind the Spaces proxy.
FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

RUN useradd --create-home --uid 1000 indicdoc
COPY --chown=indicdoc:indicdoc . .
# DocLayout-YOLO weights (Zenodo, checksum verified) and tessdata_best for the 12 languages
RUN python scripts/download_models.py && chown -R indicdoc /app/outputs

RUN mkdir -p /tmp/Ultralytics && chown indicdoc /tmp/Ultralytics
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics MPLCONFIGDIR=/tmp/matplotlib PYTHONUNBUFFERED=1 \
    YOLO_AUTOINSTALL=False YOLO_OFFLINE=True
USER indicdoc
EXPOSE 8501
# The Spaces page shows the app in an iframe through a proxy: uploads need XSRF protection and CORS off there
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--server.enableXsrfProtection=false", "--server.enableCORS=false", \
     "--browser.gatherUsageStats=false", "--server.maxUploadSize=20"]
