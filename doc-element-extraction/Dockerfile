# ============================================================
# Dockerfile for Arabic Financial Document Extraction App
# Target: NVIDIA GB10 (aarch64, CUDA 13.0 driver)
#
# PyTorch cu130 wheels bundle their own CUDA runtime libs,
# so we only need the NVIDIA runtime base for driver compat.
# ============================================================
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# ---- System Dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-ara \
    libmagic-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libpq-dev \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# ---- Tesseract Arabic Configuration ----
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata
ENV OMP_NUM_THREADS=1

# ---- Working Directory ----
WORKDIR /app

# ---- Python Dependencies ----
# CRITICAL: Install PyTorch + torchvision with CUDA 13.0 support FIRST
# from the official PyTorch cu130 index (supports aarch64/GB10).
# Then install the rest of the requirements.
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel uv && \
    pip3 install --no-cache-dir \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cu130 && \
    uv pip install --no-cache-dir --system -r requirements.txt

# ---- Application Code ----
COPY src/ ./src/

# ---- Expose Streamlit Port ----
EXPOSE 8501

# ---- Health Check ----
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# ---- Entrypoint ----
ENTRYPOINT ["streamlit", "run", "src/app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.maxUploadSize=200", \
    "--theme.base=dark"]
