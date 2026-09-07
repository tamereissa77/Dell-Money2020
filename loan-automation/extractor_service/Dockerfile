FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download EasyOCR models (optional, but speeds up first boot)
# EasyOCR downloads to ~/.EasyOCR/model by default
RUN python -c "import easyocr; easyocr.Reader(['ar', 'en'], gpu=False)"

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8014"]
