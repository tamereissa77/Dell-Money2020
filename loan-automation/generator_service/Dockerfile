FROM python:3.10-slim

# Install chromium for html2image
RUN apt-get update && apt-get install -y \
    chromium \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Set env var for html2image to find chromium
ENV CHROME_BIN=/usr/bin/chromium

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
