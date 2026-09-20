# Booth UI + scoring API. Talks to Triton over the compose network.
#
# confluent-kafka is installed here rather than in docker/svc so the whole v2
# stack shares one pip layer: docker/svc builds FROM this image, so saving both
# for the offline bundle adds a wheel, not a second 25 GB base.
FROM nvcr.io/nvidia/cugraph/financial-fraud-training:2.0.0
RUN pip install --no-cache-dir flask "tritonclient[http]" "confluent-kafka==2.6.1"
ENV PYTHONUNBUFFERED=1 MPLBACKEND=Agg
WORKDIR /app
ENTRYPOINT ["python","/app/demo_server.py"]
