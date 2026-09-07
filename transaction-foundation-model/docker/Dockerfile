# Base on the image proven to run RAPIDS cudf + CUDA 13 on GB10 (from the
# portfolio demo). The NeMo container's cudf segfaults on sm_121 via a ctypes
# CUDA probe, so we use this instead — the notebooks only need transformers,
# cudf and xgboost, not NeMo itself (pretraining is skipped; the repo ships
# a trained checkpoint).
FROM portopt-bench:latest
WORKDIR /workspace
ENV PYTHONUNBUFFERED=1 HF_HOME=/workspace/.hf TRANSFORMERS_OFFLINE=0
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
      "transformers>=4.53" safetensors accelerate \
      nbconvert ipykernel jupyter-client umap-learn
