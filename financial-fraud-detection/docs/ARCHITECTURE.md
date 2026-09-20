# v2 architecture

Stage 1 of the v2 build. Everything below runs on one GB10, air-gapped at run
time, native arm64.

## Stage 1 (built)

```
txn-generator ──▶ redpanda ──▶ scoring-svc ──▶ redpanda ──▶ demo UI
   :8091          txn.raw        (Triton)       txn.scored     :8090
     │            txn.truth ─────────────────────────────────────▶
     │                        (ground truth: UI only)
     └── control API: rate, mode, inject, pause/resume
```

## Broker: Redpanda, not Apache Kafka

Single binary, no ZooKeeper or KRaft ceremony, native arm64, and a much smaller
footprint on a desktop box. It speaks the Kafka API, so every client is stock
`confluent-kafka` and swapping in Apache Kafka needs no code change.

**Say "Kafka-compatible event stream", never "Apache Kafka".** The distinction
is small technically and large contractually.

Host ports: **19092** (Kafka API), **19644** (admin). Inside the compose network
it is `redpanda:9092`. Triton already holds 8000–8002 and 8080 belongs to
`openshell-gateway`, so the broker was kept well clear of both.

## Topics

| Topic | Partitions | Retention | Key | Purpose |
|---|---|---|---|---|
| `txn.raw` | 6 | 1 h | card/account id | incoming transactions |
| `txn.truth` | 6 | 1 h | txn id | ground-truth labels, UI only |
| `txn.scored` | 6 | 6 h | card/account id | score + versions + latencies |
| `screening.alerts` | 3 | 24 h | alert id | incumbent stub output (stage 2) |
| `alerts.ranked` | 3 | 24 h | alert id | model-prioritized queue (stage 2) |
| `case.events` | 3 | 7 d | case id | append-only case state (stage 3) |

Topics are created explicitly by `redpanda-init` rather than auto-created, so
partition counts and retention are declared and reviewable.

**Partitioning rationale.** Keyed by card/account id, so every transaction for
one card lands on one partition and per-entity ordering is preserved — which
matters because the GNN reasons over per-card sequences. Six partitions on the
hot path gives parallelism headroom (a second scoring replica can take three
partitions) without splitting a card across partitions. Three is enough for the
alert-side topics, whose volume is orders of magnitude lower.

## Label isolation — structural, not conventional

A risk team will ask whether the model can see the answer. The build makes the
answer verifiable rather than verbal:

1. Labels go to `txn.truth`, a **different topic**. `scoring-svc` subscribes to
   `txn.raw` only.
2. `docker-compose.yml` mounts **individual feature CSVs** into `scoring-svc` —
   `user.csv`, `transaction.csv`, `merchant.csv` and the three masks.
   `transaction_label.csv` is **not mounted**, so it does not exist in that
   container's filesystem.
3. `scoring.py` asserts this at startup and refuses to run if the label file is
   ever present.

## Services

### `txn-generator` (:8091)

Replays the TabFormer test set **in temporal order, never shuffled** — the
card/merchant/time relationships are what the GNN reasons over.

Deterministic: fixed seed (`SEED`, default 20260920), so a rehearsed demo
repeats exactly.

Fraud-rate modes, which the UI must display at all times:

| Mode | Rate | Note |
|---|---|---|
| `realistic` | 0.122 % | the true TabFormer base rate |
| `demo` | 4 % | elevated so something happens on screen within 30 s |

The test set itself is rebalanced to ~8.09 % fraud, so **neither mode is a
straight replay** — both resample against their target rate. `/status` reports
`configured_fraud_rate`, `emitted_fraud_rate`, `true_base_rate` and
`test_set_rate` side by side so the honesty claim is checkable from the API.

Control API: `GET /status`, `POST /rate`, `POST /mode`, `POST /inject/{scenario}`,
`POST /pause`, `POST /resume`, `GET /health`.

Scenarios: `merchant-city-anomaly` is implemented (it reproduces the Rome
$98.04 case from the v1 demo). The other four — `geo-velocity`, `cnp-burst`,
`account-takeover`, `mule-fanin-fanout` — are declared and return HTTP 501
until Stage 2, so the API surface and the UI listing are already stable.

### `scoring-svc` (:8092)

Consumes `txn.raw` in micro-batches (`BATCH_MAX` 256, `BATCH_WAIT_MS` 40),
rebases each batch onto a dense subgraph, calls Triton, publishes `txn.scored`.

**Subgraph rebasing** is the one piece of real logic. Triton is handed only the
nodes a batch touches, with edge indices remapped to positions within those
arrays rather than the global graph. Verified against v1: across sampled
transactions the streamed score matches the v1 batch score to **4.6e-07**, which
is float noise, not a logic difference.

Every scored record carries `model_version`, `feature_version` and
`data_version` so the Stage 3 audit trail can reproduce a decision.

**Two latency numbers, never conflated:**

- `model_latency_ms` — Triton inference, per transaction within its batch
- `e2e_latency_ms` — produced → scored, which is the number a bank cares about

### demo UI (:8090)

`FEED_SOURCE` selects the feed:

- `file` (**default**) — v1 behaviour, walks the pre-scored test set. The
  original three-minute booth demo is unchanged.
- `kafka` — consumes `txn.scored` into a bounded ring (`STREAM_MAX`, 2000).
  Bounded deliberately: at 5,000 TPS an unbounded buffer is a memory leak with
  a countdown, and the UI only renders the tail.

`GET /api/stream` reports `connected`, `degraded`, `buffer`, `stale_ms` and both
latency numbers — the inputs the Stage 5 status bar needs, and the basis for the
degraded banner required by acceptance criterion 13.

The stream consumer never raises fatally: if the broker dies the UI degrades to
a stale tail with a visible banner rather than white-screening.

## Images

`ffd-svc` is built `FROM ffd-demo`, which already carries numpy, pandas,
tritonclient and (now) confluent-kafka. So the v2 services add **no meaningful
layers**, and `docker save` of the stack does not duplicate a 25 GB base — which
matters for the Stage 5 offline bundle.

| Image | Size | Arch |
|---|---|---|
| `ffd-triton` | 36.3 GB | arm64 |
| `ffd-demo` | 25.5 GB | arm64 |
| `ffd-svc` | = ffd-demo + wheel | arm64 |
| `redpanda:v24.2.7` | ~400 MB | arm64 (manifest verified) |

## Measured throughput ceiling

| Load | Produced | Scored | e2e worst | e2e mean |
|---|---|---|---|---|
| 5,000 TPS (steady, 60 s) | 4,949 | 4,949 | 74–94 ms | 53–65 ms |
| 7,000 TPS (600 s soak) | 6,911 | 6,912 | — | — |

**The pipeline sustains ~6,900 TPS**, comfortably past the 5,000 TPS
requirement. One batch of 256 costs ~27 ms of model time, implying a Triton
ceiling near 9,500 TPS, so the remaining headroom is in the single-threaded
consume/decode/produce loop.

What it lacks is capacity to **drain** a backlog fast: after a `scoring-svc`
restart under a 7,000 TPS load, lag reached 109,548 and e2e worst hit 13.6 s,
taking ~7 minutes at 5,000 TPS to clear. Graceful — no crash, no loss, `errors`
0 — but visible for minutes. A second replica on three of the six partitions is
the fix, which is why six were chosen.

Two instrumentation notes, learned the hard way:

- `e2e_latency_ms` is the **worst** produce→scored in a batch. Taking it from
  the newest message hides lag completely (~30 ms while 16 k messages behind).
- `rpk` `TOTAL-LAG` includes up to `auto.commit.interval.ms` (default 5 s) of
  already-processed messages — ~25,000 at 5,000 TPS. Apparent lag under roughly
  that is bookkeeping, not backlog.
