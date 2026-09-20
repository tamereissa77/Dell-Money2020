# Stage-1 pipeline services (txn-generator, scoring-svc).
#
# FROM the demo image on purpose: it already carries numpy, pandas,
# tritonclient and confluent-kafka, so this adds no layers of consequence and
# `docker save` of the stack does not duplicate a 25 GB base.
FROM ffd-demo:latest
ENV PYTHONUNBUFFERED=1
WORKDIR /svc
ENTRYPOINT ["python"]
