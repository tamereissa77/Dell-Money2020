# ── CUDA 12.8 + cuDNN runtime base (matches RTX 5090 / sm_120) ─────────────
# Build context: ./nbe_finetune  (set in docker-compose so local_llm_client.py is reachable)
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        libgl1 libglib2.0-0 git \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements from demo/ subfolder (relative to build context nbe_finetune/)
COPY demo/requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
        torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
    && pip install -r requirements.txt

# Copy local_llm_client.py (sits at nbe_finetune/ root)
COPY local_llm_client.py .

# Copy the demo API files
COPY demo/ .

EXPOSE 8005

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8005"]
