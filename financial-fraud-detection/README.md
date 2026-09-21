# Financial crime operations — v2

An end-to-end financial-crime pipeline running entirely on one NVIDIA GB10,
air-gapped at run time, native arm64.

It began as the NVIDIA Financial Fraud Detection blueprint — a GNN plus XGBoost
scoring a static file — and v2 turns that into something a bank can recognise:
streaming ingestion, an incumbent screening engine whose queue is re-ranked
rather than replaced, case management with a non-bypassable approval gate, a
hash-chained audit trail, and a grounded bilingual narrative for the
investigator.

**Read [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) before quoting any accuracy
number from this demo.** The public benchmark it ships with contains a label
leak, and the honest figure is not the one in the blueprint's README. That
document leads with it.

---

## Quick start

From the Money20/20 launcher (`~/APPS/Money2020`):

```bash
make fraud-v2      # full pipeline
make fraud-v1      # the original booth demo, nothing else running
make fraud-which   # which variant is up
```

Or from this directory:

```bash
make up-v2 && make prewarm
```

Then open **one URL**:

```
http://<host>:8090
```

Everything is behind that origin — live detection, screening comparison, cases,
entity graph, audit trail. No other port needs to be typed.

### The two variants

| | Containers | GPU | What it is |
|---|---|---|---|
| **v1** | 2 | ~600 MiB | the original booth demo: Triton + UI over a pre-scored file |
| **v2** | 10 | ~21.6 GB | the full pipeline, including a 9B language model |

The GB10 is shared, so a v1 audience should not be paying for v2's footprint.
`up-v1` tears the v2 services down rather than stopping them — *stopped but
present* still holds GPU on this box.

`make prewarm` is **not optional**. The Triton image ships a CUDA 12.6 torch
whose arch list stops at `compute_90`, so on sm_121 the driver JIT-compiles PTX
on first use: the first explanation costs **5.31 s** cold against **3.26 s**
warm, and the acceptance bar is 3.5 s. Prewarm takes 15 s.

---

## Architecture

```
txn-generator ─▶ redpanda ─▶ scoring-svc ─▶ redpanda ─▶ demo UI
   :8091         txn.raw      (Triton)      txn.scored    :8090
     │           txn.truth ──────────────────────────────────▶ (UI only)
     │                │
     │                ▼
     │          screening-stub ─▶ screening.alerts ─┐
     │             :8093                            ├─▶ alert-svc ─▶ alerts.ranked
     │                                 txn.scored ──┘      :8094
     │                                                       │
     │                                                       ▼
     │                                                  case.events
     │                                                       │
     └── control API                                         ▼
         rate · mode · inject · pause              audit-svc :8095
                                                   (hash-chained, append-only)

                        copilot-svc :8096 ─▶ copilot-llm :8097
                        (draft narrative)      (Nemotron 9B, vLLM)
```

### Services

| Service | Port | Role |
|---|---|---|
| `demo` | **8090** | the UI, and the only origin the browser talks to |
| `txn-generator` | 8091 | replays TabFormer as a live stream; presenter control API |
| `scoring-svc` | 8092 | micro-batches off the stream, scores via Triton |
| `screening-stub` | 8093 | stands in for the bank's certified screening engine |
| `alert-svc` | 8094 | joins alerts with scores, re-ranks, owns the case lifecycle |
| `audit-svc` | 8095 | append-only hash-chained record; evidence-pack export |
| `copilot-svc` | 8096 | grounded draft narrative, English and Arabic |
| `copilot-llm` | 8097 | Nemotron-Nano-9B-v2-FP8 on vLLM |
| `triton` | 8000–8002 | the GNN + XGBoost model server |
| `redpanda` | 19092 / 19644 | the event stream |

Ports 8091–8097 are available for debugging, but the UI needs none of them: the
demo server proxies every service. **8080 is out of bounds** — it belongs to an
unrelated service on this machine.

### The broker

**Redpanda, not Apache Kafka.** Single binary, no ZooKeeper or KRaft ceremony,
native arm64, much smaller on a desktop box. It speaks the Kafka API, so every
client is stock `confluent-kafka` and swapping in Apache Kafka needs no code
change.

Say *"Kafka-compatible event stream"*, never *"Apache Kafka"*. The distinction
is small technically and large contractually.

| Topic | Partitions | Retention | Key |
|---|---|---|---|
| `txn.raw` | 6 | 1 h | card/account id |
| `txn.truth` | 6 | 1 h | txn id |
| `txn.scored` | 6 | 6 h | card/account id |
| `screening.alerts` | 3 | 24 h | alert id |
| `alerts.ranked` | 3 | 24 h | alert id |
| `case.events` | 3 | 7 d | case id |

Keyed by card so per-entity ordering holds — the GNN reasons over per-card
sequences. Six partitions on the hot path leaves room for a second
`scoring-svc` replica to take three of them.

---

## The things worth defending

### Label isolation is structural

A risk team will ask whether the model can see the answer. The build makes that
checkable rather than verbal:

1. Labels go to `txn.truth`, a **different topic**; `scoring-svc` subscribes to
   `txn.raw` only.
2. `docker-compose.yml` mounts the six feature CSVs into `scoring-svc`
   **individually** — `transaction_label.csv` does not exist in that container's
   filesystem.
3. `scoring.py` asserts that at startup and refuses to run if it ever appears.

### The incumbent is never bypassed

`screening-stub` consumes `txn.raw` directly, sees no model score, and cannot be
disabled from upstream. The model never writes to `screening.alerts`.
`alert-svc` only reorders what the incumbent raised — every alert is preserved,
and its original rank is recoverable.

Suppression is capped at **10%** of the queue and keyed on *(rule, merchant)* —
a pattern an investigator would recognise — never on a whole rule. Every
suppression carries a written reason and stays retrievable. An earlier version
keyed on the rule alone and suppressed 80% of the queue, which wins the
comparison by hiding alerts rather than by ranking them.

### The approval gate is enforced in the service, not the UI

A UI control can be bypassed with `curl`. These cannot:

| Attempt | Result |
|---|---|
| close with no actor | 400 |
| close skipping the workflow | 409 |
| close as the person who submitted it | 403 — four eyes |
| close with an invalid disposition | 400 |

Lifecycle: `new → assigned → investigating → pending-approval → closed`.

### The audit log is append-only by construction

Records reach it **only** from `case.events`. `audit-svc` has no write route at
all — `POST/PUT/DELETE/PATCH /records` return 405, and `/append` does not exist.
Each record carries the hash of its predecessor, and `/verify` names the first
break rather than returning a boolean.

### The copilot is verified, not trusted

Every `[n]` in a generated narrative must exist in the citation set; a
hallucinated reference is a hard reject. It must cite something, and it must
pass a guard that blocks filing language. Any failure falls back to a
deterministic template narrator — measured at **0.0 s**.

The guard matches the *filing sense* of SAR/STR only. **"SAR" is the Saudi
Riyal**, and the policy corpus quotes thresholds in it; a bare keyword match
rejected perfectly good narratives.

---

## Two latency numbers, never conflated

- `model_latency_ms` — Triton inference, per transaction within its batch
- `e2e_latency_ms` — produced → scored, the number a bank cares about, reported
  as the **worst** case in a batch

Taking e2e from the newest message in a batch hides consumer lag completely: it
read 32 ms while sitting 16,000 messages behind.

---

## Measured on this box

| | |
|---|---|
| Sustained throughput | **6,911 produced / 6,912 scored TPS** over 600 s, break-even |
| Steady state at 5,000 TPS | e2e worst 74–94 ms, mean 53–65 ms |
| Batch scoring | 575,775/s (25,803 rows in 0.0448 s) |
| Explanation, warm | 3.256 s — per-decision Shapley, not batch-averaged |
| Ranking lift, top 500 | 11.8× to 47× depending on the window |
| Power-cycle to usable | 13 s (43 s including the LLM) |
| Broker loss recovery | 5 s, unattended |

The lift is a **range, not a constant** — it depends on how many frauds are in
the current queue window. Quote it with the window it came from.

---

## Repository layout

```
app/            demo server (Flask) and the UI
  static/         app.html   the v2 shell - one URL, all views
                  index.html the v1 booth page, untouched
                  compare.html, graph.html
services/       the pipeline
  common/         topics, schema, Kafka wrappers
  txn_generator/  replay + scenarios + presenter API
  scoring_svc/    micro-batch scoring against Triton
  screening_stub/ the incumbent
  alert_svc/      ranking, suppression, case lifecycle
  audit_svc/      hash-chained log, evidence packs
  copilot_svc/    grounded narrative, template + LLM
models/         the SERVED Triton repo (carries the Shapley patch)
training/       training output; the unpatched upstream copy lives here
src/            blueprint preprocessing (parameterised - see LIMITATIONS)
data/policy/    12 synthetic policy documents the narrative cites
docs/           see below
```

### Documentation

| Document | What it answers |
|---|---|
| [`LIMITATIONS.md`](docs/LIMITATIONS.md) | **read first** — the label leak and what survives it |
| [`ACCEPTANCE.md`](docs/ACCEPTANCE.md) | every acceptance criterion, measured |
| [`ARCHITECTURE.md`](docs/ARCHITECTURE.md) | design decisions and why |
| [`OPERATIONS.md`](docs/OPERATIONS.md) | failure modes, recovery, the morning checklist |
| [`DEMO-GUIDE.md`](docs/DEMO-GUIDE.md) | how to present it |
| [`BASELINE.md`](docs/BASELINE.md) | the v1 state this was built from |
| [`UPSTREAM_DELTA.md`](docs/UPSTREAM_DELTA.md) | what is blueprint, what is local |
| [`PATCHES.md`](docs/PATCHES.md) | modifications to upstream artefacts |

---

## Operations

**Restart the stack each morning of a multi-day event.** Triton's inference
latency drifts from ~27 ms to several seconds over a long run, and an unattended
18-hour run throttled the pipeline from ~6,900 TPS to 26 TPS **while reporting
no errors**. `OPERATIONS.md` has the checklist and the four causes.

Two traps that cost time:

- **`docker compose up -d` does not reload bind-mounted code.** Compose only
  recreates when *configuration* changes. Use `docker compose restart <svc>`.
- **`FEED_SOURCE` is fixed at container create time**, and `.env` defaults to
  `file`. A bare `compose up -d demo` silently reverts the UI to the v1 replay.
  Use `make stream` / `make classic` / `make mode`.

### Offline

```bash
make offline-save     # 49 GB bundle, ~42 min
make offline-verify   # sha256 manifest, 24 s
make offline-load     # ~11.5 min  (FORCE=1 to replace an existing dataset)
```

The bundle carries images, the model weights **and** the gitignored
preprocessed dataset. The weights are in a named volume rather than an image
layer — vLLM stores them in the HuggingFace cache, and mounted at the wrong path
9.6 GB of weights land in the container's writable layer where `docker save`
cannot see them.

---

## Acceptance status

| Status | Criteria |
|---|---|
| **Pass** | 1, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15 |
| **Fails as specified** | 2 — `offline-load` measured at 693 s against a 90 s budget; recommendation in `ACCEPTANCE.md` |
| **Not satisfiable on this data** | 8 — see `LIMITATIONS.md` |
| **Untested** | 3 — cable unplugged; do it on the booth machine |

---

## Licence and provenance

Built on the NVIDIA AI Blueprint *financial-fraud-detection*. Dataset is IBM
TabFormer (Apache 2.0), public and synthetic-adjacent. The local delta to
upstream model code is **two lines in `model.py` and five in `config.pbtxt`** —
everything else local is a new file. See `UPSTREAM_DELTA.md`.

This platform performs **no sanctions or AML screening**. It prioritises and
explains the output of a system that does.
