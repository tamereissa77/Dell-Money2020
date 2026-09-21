"""Shared plumbing for the v2 streaming pipeline.

Topic names, message schema helpers, and thin Kafka producer/consumer wrappers.
The broker is Redpanda, which speaks the Kafka API — everything here is stock
`confluent_kafka`, so swapping in Apache Kafka needs no code change.
"""
import json, os, time, socket

BROKER = os.environ.get("BROKER", "redpanda:9092")

# Topic names are constants, not literals scattered through the services, so the
# label-isolation rule below is greppable and provable.
T_RAW = "txn.raw"
T_TRUTH = "txn.truth"          # ground truth - UI only, never the scoring path
T_SCORED = "txn.scored"
T_SCREENING = "screening.alerts"
T_RANKED = "alerts.ranked"
T_CASES = "case.events"

# Versions stamped onto every scored record so the audit trail can reproduce a
# decision. Overridable from compose; defaults describe the shipped artefacts.
MODEL_VERSION = os.environ.get("MODEL_VERSION", "prediction_and_shapley_np:1")
FEATURE_VERSION = os.environ.get("FEATURE_VERSION", "tabformer_np:v1")
DATA_VERSION = os.environ.get("DATA_VERSION", "tabformer-test-gnn:v1")


def producer(client_id="ffd"):
    from confluent_kafka import Producer
    return Producer({
        "bootstrap.servers": BROKER,
        "client.id": f"{client_id}-{socket.gethostname()}",
        "linger.ms": 5,              # small batches: latency matters more than throughput here
        "compression.type": "lz4",
        "enable.idempotence": True,  # no duplicates if a broker hiccup retries
    })


def consumer(group, topics, offset="latest"):
    from confluent_kafka import Consumer
    c = Consumer({
        "bootstrap.servers": BROKER,
        "group.id": group,
        "auto.offset.reset": offset,
        "enable.auto.commit": True,
        "session.timeout.ms": 10000,
        # librdkafka prefetches up to 1 GB PER PARTITION by default. With three
        # topics and 15 partitions a consumer that falls behind will happily
        # buffer ~15 GB of RAM before anything complains - alert-svc reached
        # 19.65 GiB this way during an overnight run. Cap it: falling behind
        # should show up as lag, which is visible, not as memory, which is not.
        "queued.max.messages.kbytes": 65536,
        "fetch.max.bytes": 52428800,
    })
    c.subscribe(topics if isinstance(topics, list) else [topics])
    return c


def send(prod, topic, key, value):
    """Fire-and-forget publish. Key sets the partition, so all traffic for one
    card lands on one partition and stays in order."""
    prod.produce(topic, key=str(key).encode(), value=json.dumps(value).encode())
    prod.poll(0)


def decode(msg):
    return json.loads(msg.value().decode())


def wait_for_broker(timeout=120):
    """Block until the broker answers. Compose health checks cover the normal
    case; this catches the cold-start race where a service beats Redpanda up."""
    from confluent_kafka.admin import AdminClient
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            md = AdminClient({"bootstrap.servers": BROKER}).list_topics(timeout=5)
            if md.brokers:
                return True
        except Exception as e:                      # broker not up yet
            last = e
        time.sleep(1)
    raise RuntimeError(f"broker {BROKER} unreachable after {timeout}s: {last}")


def now_ms():
    return time.time() * 1000.0
