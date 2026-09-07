# Triton + GPU XGBoost + PyG + Captum, for serving the fraud GNN with Shapley values.
# NOTE: this base ships CUDA 12.6 PyTorch (arch list stops at compute_90). On GB10
# (sm_121) kernels are JIT-compiled from PTX at first launch - ~85s unless the CUDA
# JIT cache is warm. The compose file mounts a persistent cache; run `make prewarm`
# after a cold boot.
FROM nvcr.io/nvidia/tritonserver:25.04-py3

WORKDIR /opt
ENV DEBIAN_FRONTEND=noninteractive CONDA_DIR=/opt/miniconda
ENV PATH=${CONDA_DIR}/bin:${PATH}
SHELL ["/bin/bash","-o","pipefail","-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
      wget curl git build-essential cmake libopenblas-dev libssl-dev libtinfo-dev \
 && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    ARCH="$(uname -m)"; \
    if [ "$ARCH" = "aarch64" ]; then \
      URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh; SUB=linux-aarch64; \
    else \
      URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh; SUB=linux-64; \
    fi; \
    wget -q "$URL" -O miniconda.sh; bash miniconda.sh -b -p "$CONDA_DIR"; rm miniconda.sh; \
    conda config --set subdir "$SUB"; conda config --set always_yes true; \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main; \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r; \
    conda update -n base -c defaults conda

ENV PYTHONPATH="$CONDA_DIR/lib/python3.12/site-packages:$PYTHONPATH" \
    LD_LIBRARY_PATH="$CONDA_DIR/lib:$LD_LIBRARY_PATH" \
    CONDA_OVERRIDE_CUDA=12.8 CONDA_ALWAYS_YES=true

RUN conda install python=3.12
RUN conda install --override-channels -c nvidia -c conda-forge \
      pytorch=2.7.0=*cuda126* py-xgboost=3.0.2=*cuda128* ncurses
RUN conda install --override-channels -c conda-forge cupy=13.4.1
RUN $CONDA_DIR/bin/pip install --no-cache-dir \
      torch-geometric==2.6.1 captum==0.7.0 \
      --extra-index-url https://data.pyg.org/whl/torch-2.7.0+cu126.html
RUN conda install -y --override-channels -c rapidsai -c nvidia -c conda-forge cudf=25.04 python=3.12
