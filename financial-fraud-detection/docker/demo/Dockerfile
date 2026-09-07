# Booth UI + scoring API. Talks to Triton over the compose network.
FROM nvcr.io/nvidia/cugraph/financial-fraud-training:2.0.0
RUN pip install --no-cache-dir flask "tritonclient[http]"
ENV PYTHONUNBUFFERED=1 MPLBACKEND=Agg
WORKDIR /app
ENTRYPOINT ["python","/app/demo_server.py"]
