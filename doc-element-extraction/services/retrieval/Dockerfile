# ============================================================
# Dockerfile — Retrieval Agent Microservice
# ============================================================
# Lightweight Python image with Flask + gunicorn.
# No GPU required — this service only talks to other services
# (Ollama, Qdrant, PostgreSQL, ArangoDB, MinIO, Sentence-Transformers).
# ============================================================

FROM python:3.11-slim

# System deps for psycopg2-binary
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libpq-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY retrieval_agent.py /app/
COPY app.py /app/

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8100/health/simple || exit 1

EXPOSE 8100

# Run with gunicorn: 2 workers, 4 threads each
CMD ["gunicorn", \
     "--bind", "0.0.0.0:8100", \
     "--workers", "2", \
     "--threads", "4", \
     "--timeout", "300", \
     "--graceful-timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "app:app"]
