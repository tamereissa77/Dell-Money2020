# One-shot pipeline tooling: preprocessing (cuDF), training launch, smoke tests.
FROM nvcr.io/nvidia/cugraph/financial-fraud-training:2.0.0
RUN pip install --no-cache-dir category_encoders "tritonclient[http]"
ENV PYTHONUNBUFFERED=1 MPLBACKEND=Agg
WORKDIR /work
ENTRYPOINT ["/bin/bash","-lc"]
