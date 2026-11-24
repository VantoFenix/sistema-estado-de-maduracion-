FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_NUM_INTRAOP_THREADS=1 \
    TF_NUM_INTEROP_THREADS=1 \
    PORT=8000

WORKDIR /app

COPY requirements-web.txt .
COPY src/ ./src/
COPY api/ ./api/
COPY artifacts/ ./artifacts/
COPY frontend/index.html ./frontend/index.html
RUN pip install --no-cache-dir -r requirements-web.txt

EXPOSE 8000

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
