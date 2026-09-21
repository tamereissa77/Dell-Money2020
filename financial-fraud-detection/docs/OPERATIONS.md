# Operations — failure modes and recovery

Everything here was found by leaving the v2 stack running unattended for 18
hours at 2,000 TPS. None of it shows up in a short test, and every one of them
degrades *silently* — health checks stayed green throughout.

## The headline

**The pipeline throttled itself from ~6,900 TPS to 26 TPS over 18 hours and
reported no errors.** `/health` returned `ok`, `errors` stayed 0, and the UI
kept serving. The only visible symptom was end-to-end latency climbing to 97
seconds and 19% of alerts failing to join.

If this stack is left running across a multi-day event, **restart it each
morning**. The causes below are fixed, but long-run degradation in Triton
itself is not.

---

## 1. Missing index on the backfill path (fixed)

**Symptom:** `alert-svc` pegged at 100% CPU, scoring starved to 26 TPS.

**Cause:** label backfill runs `UPDATE alerts SET label=? WHERE txn_id=?` and
`txn_id` had no index. At 742,000 rows a single lookup cost **61 ms**, and the
backfill issues 200 of them roughly twice a second — about 12 seconds of CPU
work arriving every 0.5 s. It degrades as the table grows, which is why it took
hours to become obvious.

**Fixed:** `ix_alerts_txn` and `ix_alerts_incrank`. Lookup dropped 61 ms → ~0 ms.

**Lesson:** any query on the ingest path needs an index before the table is
large, not after. `ORDER BY score` was indexed and stayed at 0 ms throughout;
`ORDER BY incumbent_rank` was not and cost 75 ms.

## 2. Unbounded queue table (fixed)

**Symptom:** 741,975 rows after 18 hours, every unindexed query degrading with it.

**Fixed:** `ALERT_MAX_ROWS` (default 200,000), pruned every 60 s by
`incumbent_rank`. The alerts table is a live queue, not a system of record —
Stage 3's audit trail is the durable artefact.

## 3. librdkafka prefetch (fixed)

**Symptom:** `alert-svc` at **21.77 GiB** RSS.

**Cause:** librdkafka prefetches up to **1 GB per partition** by default. Three
topics × 15 partitions ≈ 15 GB of buffer before anything complains.

**Fixed:** `queued.max.messages.kbytes = 65536` (64 MB/partition) and
`fetch.max.bytes = 50 MB` in `common/pipeline.py`. Memory fell to 112 MiB.

**Lesson:** a consumer falling behind should surface as **lag**, which is
visible, not as **memory**, which is not.

## 4. Consumer groups replay their whole backlog on restart

**Symptom:** after a restart, `alert-svc` reported `joined: 0` and `unmatched`
climbing into the millions. Group lag reached **88.5 M** (alert-svc) and
**126 M** (demo-ui).

**Cause:** `auto.offset.reset: latest` applies **only when a group has no
committed offset**. A group that has run before resumes from its committed
position and grinds through everything since. Those alerts' scores had already
aged out of `txn.scored` (6 h retention), so nothing could match — the join was
structurally dead until the backlog cleared.

**Recovery:**

```bash
docker compose stop alert-svc screening-stub demo
sleep 25                                   # let session.timeout (10s) expire
for g in alert-svc screening-stub demo-ui; do
  docker exec ffd-redpanda rpk group seek $g --to end -X brokers=localhost:9092
done
docker compose start screening-stub alert-svc demo
```

The `sleep` matters: `rpk group seek` refuses with `INVALID_OPERATION: seeking a
non-empty group is not allowed` while a member is still registered. Our SIGTERM
handler calls `os._exit(0)`, which does not leave the group cleanly, so the
session timeout has to expire.

## 5. Triton degrades over long runs (NOT fixed)

**Symptom:** `model_latency_ms` drifted from 27 ms to **6,700 ms** for the same
batch size. Triton sat at 100% CPU logging
`evhtp.c:3130 ODDITY, resuming when not paused?!? (Connection reset by peer)` in
a loop.

**Recovery:** restart Triton, then `make prewarm`. Verified immediately after:

| Batch | Latency | Implied ceiling |
|---|---|---|
| 32 | 26.6 ms | 1,202 TPS |
| 256 | 26.6 ms | 9,620 TPS |
| 1024 | 27.1 ms | 37,844 TPS |

**This has no fix in the build** — it is upstream behaviour under a sustained
136-million-inference load. Restart Triton before any long demo session.

**Useful side finding:** latency is essentially **fixed overhead**, flat from
batch 32 to 1024. Throughput therefore scales almost linearly with batch size,
so `BATCH_MAX` is the cheapest throughput lever available — 1024 would raise the
ceiling roughly 4× over the current 256 at no latency cost.

## 6. `compose up -d` does not reload bind-mounted code

Editing a file under `services/` and running `docker compose up -d` leaves the
old code running: compose only recreates when *configuration* changes, and a
bind-mounted file is not configuration. Use `docker compose restart <svc>`.

Same class of problem as VSS resuming with a stale `HOST_IP` — see the root
`CLAUDE.md`.

---

## Morning checklist for a multi-day event

```bash
cd ~/APPS/Money2020/financial-fraud-detection
docker compose restart triton && make prewarm      # item 5 - required
docker compose restart scoring-svc alert-svc screening-stub
curl -s localhost:8092/metrics | python3 -m json.tool   # model_ms should be ~27
curl -s localhost:8094/health  | python3 -m json.tool   # joined climbing, unmatched flat
docker exec ffd-redpanda rpk group describe scoring-svc -X brokers=localhost:9092
```

`model_ms` far above ~35 ms, or `unmatched` rising while `joined` stays flat,
means one of the above has recurred.
