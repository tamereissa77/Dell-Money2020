FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py /app/

ENV MODEL_NAME="all-MiniLM-L6-v2"
ENV TRANSFORMERS_CACHE="/app/.cache"

# Pre-download the model during build
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 80

CMD ["gunicorn", "--bind", "0.0.0.0:80", "--workers", "1", "--threads", "8", "--timeout", "120", "app:app"]
