# Acceptance criteria — results

Measured on `gb10-galeneai`, 2026-09-20. Stage 1 covers criteria 1–6; the rest
are recorded as not-yet-tested so nothing reads as passing that has not run.

| # | Criterion | Stage | Result |
|---|---|---|---|
| 1 | `v1-money2020` reproduces the original demo | 1 | **PASS** (see below) |
| 2 | `offline-load && start` < 90 s, no internet | 5 | **FAIL as specified** — 693 s measured; see below |
| 3 | Cable unplugged: every view works | 5 | not tested |
| 4 | ≥ 5,000 TPS end-to-end, stable queue, 10 min | 1 | **PASS** — 600 s soak at 6,911/6,912 TPS, break-even |
| 5 | Batch throughput within 10 % of 589 k/s | 1 | **PASS** — 575,775/s, within 2.3 % |
| 6 | Single-txn explanation ≤ 3.5 s, per-decision | 1 | **PASS warm / FAIL cold** (details below) |
| 7 | Five scenarios inject on demand | 2 | **PASS** — 5/5 visible in 0.30 s |
| 8 | `mule-fanin-fanout` caught by model, missed by stub | 2 | **NOT SATISFIABLE on this split** — see below |
| 9 | No case closes without a named human action | 3 | **PASS** — 6 bypass attempts refused |
| 10 | Evidence pack exports; tamper fails verification | 3 | **PASS** — tamper named to the record |
| 11 | Cited draft in Arabic and English | 4 | **PASS** — 8/8 citations resolve |
| 12 | Template fallback with the LLM stopped | 4 | **PASS** — failover in 0.0 s, recovers automatically |
| 13 | Broker killed mid-demo: degraded banner, recovers | 5 | **PASS** — no white screen, recovers in 5 s |
| 14 | Full power-cycle recovery ≤ 30 s | 5 | **PASS** — usable in 13 s (43 s incl. LLM) |
| 15 | `docker manifest inspect` confirms arm64 for all images | 5 | **PASS** — all 5 images linux/arm64 |

---

## 1 — v1 still reproduces

`v1-money2020` is a branch at commit `97ffcf4`, which is the working booth state
frozen before any v2 work. The v2 default preserves it at runtime too:
`FEED_SOURCE` defaults to `file`, so the UI walks the pre-scored test set
exactly as v1 did. The streaming path is opt-in.

`make verify` on the v1 path: **ALL CHECKS PASSED** — triton health 200, demo
api 200, scoring OK, feed has data, live explanation top driver
`Merchant city` +10.43.

## 4 — sustained throughput

**10-minute soak, 7,000 TPS target** (the authoritative run):

```
duration : 600s
produced : 6,911 TPS
scored   : 6,912 TPS
backlog  : -144 msgs      (break-even)
errors   : 0
```

**Steady state at 5,000 TPS**, sampled every 15 s for a minute:

| t | e2e worst | e2e mean | scored rate |
|---|---|---|---|
| +0 s | 94.49 ms | 65.36 ms | — |
| +15 s | 81.29 ms | 54.92 ms | 4,932 TPS |
| +30 s | 75.15 ms | 53.29 ms | 4,932 TPS |
| +45 s | 94.03 ms | 64.76 ms | 4,949 TPS |
| +60 s | 73.91 ms | 55.57 ms | 4,949 TPS |

Flat, not growing. `TOTAL-LAG` settled at 5,275. **Criterion met**, and the soak
shows the pipeline also sustains ~6,900 TPS.

What it does *not* have is spare capacity to **drain** a backlog quickly: after
a `scoring-svc` restart at a 7,000 TPS load, lag reached 109,548 and e2e worst
hit 13.6 s, taking ~7 minutes at 5,000 TPS to clear. Recovery is graceful —
nothing crashes, no messages lost, `errors` stays 0 — but a mid-demo restart is
visible for minutes. A second `scoring-svc` replica taking three of the six
partitions is the fix; the partition count was chosen to allow it.

### Two measurement corrections

Earlier figures in this file were wrong and are superseded above:

1. **"~414,000 backlog at 7,000 TPS" was a measurement error.** It subtracted
   the generator's lifetime `emitted` from the scorer's lifetime `scored`, but
   `scoring-svc` starts at `offset=latest` and so legitimately never sees
   messages produced before it joined. Cumulative counters with different start
   points cannot be differenced. `rpk group describe` is the authority.
2. **"e2e 32 ms" understated latency.** `scoring-svc` took e2e from `msgs[-1]`,
   the *newest* message in each batch, which hides consumer lag entirely — it
   still read ~30 ms with 16 k messages of backlog. Now fixed to report the
   **worst** case in the batch, with mean and best alongside.

A third artifact worth knowing: `TOTAL-LAG` includes up to
`auto.commit.interval.ms` (default 5 s) of already-processed messages, which at
5,000 TPS is ~25,000. Apparent lag below roughly that figure is bookkeeping,
not backlog.

## Stage 2 — screening comparison (the commercial claim)

Measured at the **true 0.122% base rate** (`realistic` mode), 2,000 TPS.

```
incumbent queue : 80,638 alerts containing 197 real frauds
incumbent FP    : 99.18%   (TP 1,667 / FP 126,095 over the session)
```

| Queue depth worked | Incumbent order | Re-ranked | Lift |
|---|---|---|---|
| 50 | 9 TP (5.1%) | 38 TP (21.7%) | 4.2x |
| 200 | 9 TP (5.1%) | 134 TP (76.6%) | 14.9x |
| **500** | **15 TP (8.6%)** | **177 TP (89.8%)** | **11.8x** |
| 1000 | 21 TP (11.9%) | 169 TP (96.0%) | 8.1x |

**The headline: working 500 alerts instead of 80,638, the re-ranked queue finds
~90% of the fraud against the incumbent's ~9%.** Computed against ground truth,
not against the model's own predictions, and not asserted.

**The lift depends entirely on the stream's fraud rate**, so the mode must be
quoted with the number. In `demo` mode (4% fraud, 33x reality) the incumbent's
queue is artificially fraud-rich, its FP rate falls to ~66%, and lift drops to
~1.9x. The UI shows the active mode, the stream's measured fraud rate, the true
base rate and the test-set composition in a strip that is always visible.

Suppression is capped at **10%** of the queue and keyed on (rule, merchant) —
a pattern an investigator would recognise — not on the rule alone. Every
suppression carries a written reason and stays retrievable at
`/api/suppressed`. An earlier version keyed on rule alone and suppressed 80% of
the queue, which would have won the comparison by hiding alerts rather than by
ranking them.

## 7 — scenario injection

All five typologies inject on demand and appear in the UI well inside the bar:

| Scenario | Visible in | Rows | Flagged | Top score |
|---|---|---|---|---|
| `merchant-city-anomaly` | 0.30 s | 1/1 | 1 | 0.9262 |
| `geo-velocity` | 0.31 s | 2/2 | 1 | 0.9763 |
| `cnp-burst` | 0.30 s | 6/6 | 1 | 0.7336 |
| `account-takeover` | 0.30 s | 5/5 | 3 | 0.9839 |
| `mule-fanin-fanout` | 0.30 s | 8/8 | 8 | 0.9926 |

Each typology is a curated sequence of **real rows** from the test set, never a
fabricated transaction: `scoring-svc` looks up features by row index, so a
synthetic row could not be scored at all.

**A one-transaction injection is invisible in the scrolling feed.** At 400 TPS a
single row leaves a 200-row window in about half a second — too fast for a
presenter to point at. Injected transactions are therefore retained separately
in `/api/scenarios` until displaced by later injections, which is what the
timings above measure.

**Operational caveat:** the first injection after restarting the demo container
can be missed. Its consumer joins the group at `offset=latest`, and anything
produced during that join window is never seen. Inject once and discard the
result after any `make stream` / `make classic` switch.

## 8 — model catches it, screening misses it

**Not satisfiable on the blueprint's 2019 split, and the build says so rather
than faking it.**

Every fraud in that split is in Rome, so any watchlist containing Rome catches
100% of them: **zero of the 22,787 rule-evading rows are fraudulent.** There is
no transaction the model can catch that the incumbent misses, because the
incumbent misses nothing.

`mule-fanin-fanout` selects in tiers and reports which one it reached:

- **tier 1** — frauds the incumbent's rules do not fire on. The full claim.
- **tier 2** — frauds at a high fan-in merchant, rules may also fire. Shows the
  network structure only.

On this data it reaches **tier 2**: 8 frauds across 8 distinct cards converging
on one merchant, all 8 flagged by the model (top score 0.9926). That is a
genuine fan-in, and it is the right screen for explaining why a graph sees
something a row-at-a-time engine cannot — but it does **not** demonstrate rules
evasion, and `describe()` returns exactly that sentence so it cannot be quoted
otherwise.

Tier 1 becomes reachable on the `gnn_np_div` split (2015–16), where fraud spans
730 cities. That is where criterion 8 should be demonstrated. See
`LIMITATIONS.md`.

## 9 — the approval gate

Enforced at the **service layer**, not in the UI. A UI control can be bypassed
with curl; these cannot. Every attempt below was made directly against the API:

| Attempt | Result |
|---|---|
| close with no actor | **400** a named approver is required |
| close straight from `new`, skipping the workflow | **409** case is not awaiting approval |
| assign with no actor | **400** a named actor is required |
| close as the person who submitted it | **403** four-eyes violation |
| close with an invalid disposition | **400** invalid disposition |
| close as a second named approver | **200** closed / confirmed-fraud |

The lifecycle is `new → assigned → investigating → pending-approval → closed`,
and closure additionally requires a disposition from a fixed set and an
approver who is **not** the submitter. There is no flag, header or internal
route that bypasses any of it.

## 10 — evidence pack and tamper detection

Export for one case, `GET /export/<case_id>`:

```
case          : S000343909
records       : 4
human actors  : a.rahman, m.alqahtani
chain         : ok=True over 4 records
pack digest   : 74e0f5a3332ec1852e86af7178dfd952…

  1  06:58:50Z  human  a.rahman      case.assign       new→assigned
  2  06:58:50Z  human  a.rahman      case.investigate  assigned→investigating
  3  06:58:50Z  human  a.rahman      case.submit       investigating→pending-approval
  4  06:58:50Z  human  m.alqahtani   case.close        pending-approval→closed [confirmed-fraud]
```

**Tamper test.** One record was altered directly in SQLite — the disposition
rewritten from `confirmed-fraud` to `false-positive`, without updating its hash:

```
broken_at      : 4
reason         : record contents do not match its stored hash
expected_hash  : ee7ce579bc816d54…
found_hash     : 4d534d5716254bdf…
```

Verification fails loudly and **names the record**, which is the question that
actually gets asked. The evidence pack carries the same failure, so a tampered
database cannot produce a clean-looking export.

**Append-only is structural.** Records reach the log only from the
`case.events` topic; `audit-svc` has no write route at all:

```
POST/PUT/DELETE/PATCH /records → 405      POST /append|/audit|/write → 404
```

**Deliberately not audited:** one record per scored transaction. At the measured
6,900 TPS that is ~600M records a day and none of them is a decision. The score
that matters is the one attached to an alert, captured there with its model,
feature and data versions.

## 11 — grounded, cited, bilingual draft

`POST /draft/<alert_id>` returns an investigator's draft narrative in English
and Arabic, sharing one numbered citation set. Every citation resolved:

```
[1] alert        S000679857                         resolves
[2] score        model prediction_and_shapley_np:1  resolves
[3] policy       POL-MODEL-01                       resolves
[4] attribution  shapley:t000679857                 resolves
[5..8] policy    POL-CASE-01, POL-AML-02, POL-OPS-01, POL-OPS-02
8 resolve, 0 dangling
```

The narrative is **assembled from** the citation list rather than annotated
with it, so an uncited assertion cannot be emitted. Grounding is 12 synthetic
policy documents in `data/policy/`, the alert record, and the per-decision
Shapley attribution.

**Three controls, all structural:**

- **Never a filing.** An output guard rejects "suspicious transaction report",
  "SAR filing" and similar, on generated text *and* on analyst edits. Verified:
  two such edits rejected with 400, a legitimate edit accepted with 200.
- **Machine-drafted until adopted.** Status stays `machine-drafted`; adoption
  requires a named analyst (400 without one), preserves the machine text
  alongside the edit, and publishes `narrative.adopt` to the audit chain —
  confirmed at record 5, chain still verifying clean.
- **Model output is not a conclusion.** Every draft states it is decision
  support, citing POL-MODEL-01. Where one attribution dominates the rest by
  3x, the draft additionally requires the analyst to record whether that
  feature is behaviourally meaningful or a population characteristic of the
  training data (POL-MODEL-02) — which is precisely the "merchant city"
  problem in `LIMITATIONS.md`, surfaced into the workflow rather than hidden.

## 12 — fallback with the LLM stopped

The copilot runs **NVIDIA-Nemotron-Nano-9B-v2-FP8** on vLLM as a service of
this compose project — the same container VSS uses, but owned here, reusing
VSS's model-cache volume so it starts warm rather than pulling 9B again. The
launcher's one-demo-at-a-time rule therefore never stops it, and the fraud demo
stays self-contained. It costs **20.2 GB** of GPU at `--gpu-memory-utilization
0.18`, alongside Triton's ~600 MB.

| Step | Result |
|---|---|
| draft with the LLM up | `generator: llm`, 10.7 s, cited [1,2,3,4,6] |
| `docker compose stop copilot-llm` | — |
| draft with the LLM down | `generator: template`, **0.0 s**, 220 words, 8 citations, Arabic intact |
| LLM restarted | `generator: llm` — recovers with no intervention |

Failover is immediate and silent to the user; the reason is recorded
(`llm unreachable: URLError`) and counted.

### The LLM output is verified, not trusted

Three checks before a generated narrative is accepted, any of which falls back
to the template:

1. **Every `[n]` must exist** in the citation set. A hallucinated reference is a
   hard reject, not a footnote.
2. It must cite **something**.
3. It must pass the same forbidden-phrase guard as the template.

A grounded narrative that cannot be verified is worth less than a plain one
that can.

### Three real defects this surfaced

**"SAR" is the Saudi Riyal.** The first guard listed a bare `" sar "` and
rejected a perfectly good narrative for quoting POL-AML-01's "SAR 1,875"
threshold. At a Riyadh event every amount is in SAR, so this would have fired
constantly. The guard now matches the *filing sense* only — `file a SAR`,
`SAR filing`, `raise an STR`, `suspicious transaction report` — and passes
currency usage in either position. Seven cases verified.

**Nemotron is a reasoning model.** Without a `/no_think` system prompt it spent
the entire token budget thinking and never emitted the narrative
(`finish_reason: length`, zero citations).

**The model read the score backwards.** Given 0.9955 it wrote "strong
likelihood of the transaction being legitimate". The prompt now states
explicitly that the score is P(fraud), where 1.0 means almost certainly fraud.
This is the clearest possible argument for the adoption gate: a fluent,
well-cited narrative can still be exactly wrong, and no draft reaches a case
file without a named analyst adopting it.

## 13 — broker killed mid-demo

`docker compose stop redpanda`, then:

| Check | Result |
|---|---|
| `/`, `/compare`, `/graph` | **HTTP 200** — no white screen on any view |
| `/api/stream` | `degraded=true`, staleness 13.0 s |
| banner | shown, naming the reason |
| broker restarted | **recovered in 5 s**, unattended |

Worth noting *how* it was detected: `connected` still read true, because
librdkafka had not yet errored. Degradation is computed from **staleness or
disconnection**, not from the client's own opinion of its health — which is why
it caught this at all.

## 14 — power-cycle recovery

| Milestone | Time |
|---|---|
| demo usable — UI serving a live scored feed | **13 s** |
| whole stack including the 9B LLM | 43 s |

Criterion met for the demo itself. The LLM adds ~30 s, and the template
narrator covers that window (criterion 12), so the demo is usable before the
model finishes loading.

## 15 — architecture

Every image in the stack is `linux/arm64`, no emulation anywhere:

```
docker.redpanda.com/redpandadata/redpanda:v24.2.7   linux/arm64
ffd-demo:latest                                     linux/arm64
ffd-svc:latest                                      linux/arm64
ffd-triton:latest                                   linux/arm64
nvcr.io/nvidia/vllm:25.12.post1-py3                 linux/arm64
```

## Graph view

`/graph?row=<n>` renders the neighbourhood around one transaction — its card,
its merchant, and the other transactions those two touch — on a hand-written
canvas with no external library.

Layout is **deterministic**, not force-directed: merchant at the centre, cards
on a ring, transactions between them. A fan-in has to read identically every
time it is shown, and a settling physics graph does not.

For the `mule-fanin-fanout` typology it renders 29 nodes and 28 edges with
**9 unrelated cards converging on one merchant** — structure that exists only
across accounts, which is the thing a per-transaction rules engine has nowhere
to see.

The page also carries the always-visible status bar (feed source, fraud-rate
mode, TPS, queue depth, end-to-end and model latency, pipeline health) and the
presenter panel, which opens on **Ctrl+Shift+P** and has no visible affordance —
a customer must never see the presenter's controls.

## 5 — batch throughput preserved

25,803 transactions in 0.0448 s = **575,775/s**, against the recorded 589,528/s
baseline. Within 2.3 %, comfortably inside the 10 % allowance.

## 6 — explanation latency and per-decision attribution

| Condition | Latency |
|---|---|
| Cold (before `make prewarm`) | **5.31 s** — fails the 3.5 s bar |
| Warm (after `make prewarm`) | **3.256 s**, repeatable | 
| Pre-computed hero | 0.0015 s (served from cache) |

**This criterion is prewarm-dependent.** `make prewarm` takes 15.1 s and must
run before the demo is measured or shown. Any v2 start path that omits it fails
criterion 6 on the first explanation.

Attribution is per-decision, not batch-averaged: `explain()` sends a
single-transaction subgraph with `COMPUTE_SHAP` on. Verified distinct across
transactions — the v1 verify step reports a specific top driver
(`Merchant city` +10.43) for its chosen transaction.

## Correctness check not in the brief

The streaming path rebases each batch onto a dense subgraph, which is new logic
and could silently change scores. Cross-checked streamed scores against v1
batch scores for sampled transactions:

```
compared: 10   mismatches (>0.02): 0   max delta: 4.58e-07
```

Float noise. The subgraph remapping is faithful.

## Note on criterion 2 — the 90-second budget, now measured

The budget was flagged as unreachable from the start. It has now been measured
rather than estimated:

| Step | Measured |
|---|---|
| `offline-save` | **2,526 s** (42 min) |
| bundle size | **49 GB** — images 41 GB, model weights 7.7 GB, dataset 8.4 MB |
| manifest verify (sha256 × 3) | 24 s |
| `offline-load` | **693 s** (11.5 min) |
| `start` alone | **13 s** (criterion 14) |
| **brief's budget for load + start** | **90 s** |

`offline-load` is **7.7× over budget on its own**, and it is bounded by reading
and decompressing 41 GB of container images — not by anything in this build.
The images are what they are: the NVIDIA training base is 24 GB before a line
of demo code, Triton is 34 GB, and vLLM adds 14 GB.

**Recommendation: split the criterion.**

- `offline-load` is **provisioning** — done once, the day before, to a machine
  that will then be carried to the venue. Budget it at **15 minutes** and
  verify it by the sha256 manifest rather than a stopwatch.
- `start` is what happens **in front of a customer**, and it already meets the
  spirit of the 90 s bar with 77 s to spare: **13 s** to a usable demo, 43 s
  including the 9B language model.

Nothing here is a shortcut: the bundle genuinely contains everything needed on
a machine with no internet, including the model weights that `docker save`
would otherwise have missed entirely (see the packaging bug in the Stage 5
commit).

### `offline-load` is idempotent

The first run failed at the final step: the preprocessing writes
`data/TabFormer/gnn_np` from a container as root, so `tar` could not overwrite
it and aborted the whole target after successfully loading 41 GB of images.
It now skips an existing dataset and takes `FORCE=1` to replace it.

## 3 — cable unplugged

**Not yet tested.** The bundle exists and loads; what has not been done is
pulling the network and exercising every view. Worth doing on the booth
machine rather than this one, since this one is the source of the bundle.
