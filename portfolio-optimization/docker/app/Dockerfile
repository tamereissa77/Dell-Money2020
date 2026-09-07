# Portfolio optimization (NVIDIA cuOpt + RAPIDS cuML) — Streamlit booth demo.
# The CUDA 13 wheels are Blackwell-native, so unlike the fraud demo there is no
# PTX JIT penalty on GB10 (sm_121) and no pre-warm step is required.
FROM nvcr.io/nvidia/pytorch:25.10-py3

WORKDIR /work
ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

COPY pyproject.toml README.md VERSION ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e ".[cuda13]"

COPY demo/requirements.txt ./demo/requirements.txt
RUN pip install --no-cache-dir -r demo/requirements.txt

COPY demo/ ./demo/
COPY .streamlit/ ./.streamlit/

EXPOSE 8501
CMD ["streamlit","run","demo/rebalancing_streamlit_app.py", \
     "--server.address","0.0.0.0","--server.port","8501"]
