# Dell Technologies × NVIDIA — Money20/20 Middle East demos

Financial-services AI demos running **entirely on a single NVIDIA GB10** — no cloud,
no network at run time. Built for Money20/20 Middle East, Riyadh, 14–16 September 2026.

Seven demos, one launcher, one machine. They share the GB10, so only one runs at a time.

---

## Demos

| Demo | Alias | What it shows |
|---|---|---|
| [`financial-fraud-detection`](financial-fraud-detection/) | `fraud` | A graph neural network + XGBoost scoring card transactions at ~575,000/sec with a per-decision Shapley explanation. **v2** adds streaming ingestion, an incumbent screening engine whose queue is re-ranked rather than replaced, case management with a non-bypassable approval gate, a hash-chained audit trail and a grounded bilingual investigator narrative. |
| [`portfolio-optimization`](portfolio-optimization/) | `portfolio` | Mean-CVaR portfolio optimisation on NVIDIA cuOpt — 18× faster than CPU at 500 assets × 50,000 scenarios, reaching the same optimum. |
| [`quant-signal-discovery`](quant-signal-discovery/) | `quant` | Three agents invent an alpha signal, write it as executable Python and backtest it on 14 years of S&P 500 prices — then read their own results and try again. Nemotron 3 Nano 30B A3B served locally; a three-iteration loop runs in ~50–75 s. |
| [`nemotron-voice-agent`](nemotron-voice-agent/) | `voice` | Speak to the GB10 and it speaks back. Nemotron ASR, a 30B Nemotron LLM and Magpie TTS as local sidecars, WebRTC to the browser. **English only** — no Nemotron LLM supports Arabic. |
| [`doc-element-extraction`](doc-element-extraction/) | `docs` | Document element extraction and knowledge-graph construction, including Arabic table extraction. Contributed by **iPulse-AI** under partnership. |
| [`loan-automation`](loan-automation/) | `loan` | Automated loan origination from Arabic identity and income documents, using a fine-tuned Qwen2.5-VL-7B vision-language model. Contributed by **Efadatek** under partnership. |
| [`vss-video-analytics`](vss-video-analytics/) | `vss` | NVIDIA Video Search and Summarisation — ask plain-language questions of recorded video. Wraps an existing on-box install rather than duplicating 3.4 GB. |

Plus a customer-facing [`showcase/`](showcase/) site (nginx, port 8088) with a page per
use case and a technical note on how to read the fraud benchmark.

---

## Running a demo

```bash
make list             # every demo, and whether it is running
make fraud            # start it — stops whatever else is running first
make fraud-prewarm    # warm GPU caches — REQUIRED after a cold boot
make fraud-status     # health, throughput, model quality
make fraud-verify     # end-to-end smoke test
make fraud-logs
make fraud-down
make stop             # stop everything
```

`make <demo>` also accepts the full directory name. **Only one demo runs at a time** — they
share a single GB10, and two model servers competing for it will run out of memory or slow
each other down. Starting one tears the others down first.

### The fraud demo has two variants

```bash
make fraud-v1       # the original booth demo: 2 containers, ~600 MiB GPU
make fraud-v2       # the full enterprise pipeline: 10 containers, ~21.6 GB GPU
make fraud-which    # which variant is up, and which feed the UI is reading
```

`make fraud` runs v2. Use `fraud-v1` when the audience only needs the original three
minutes — a v1 audience should not be paying for v2's footprint on a shared box.

> **After a cold boot, run `make <demo>-prewarm` first.** The Triton image ships a CUDA 12.6
> PyTorch whose kernels stop at `compute_90`; on GB10 (`sm_121`) they are JIT-compiled from
> PTX at first launch. Cold, the first inference costs ~85 s and the first explanation 5.3 s.
> Warm, 0.5 s and 3.3 s. Prewarm takes 15 s.

---

## Financial fraud detection

Based on the [NVIDIA AI Blueprint for Financial Fraud Detection](https://github.com/NVIDIA-AI-Blueprints/Financial-Fraud-Detection),
adapted to run natively on GB10 (arm64), then extended into an end-to-end financial-crime
pipeline. Full detail in [`financial-fraud-detection/README.md`](financial-fraud-detection/README.md).

Measured on the machine, not from a datasheet:

| | |
|---|---|
| Batch scoring | **575,775 transactions/sec** (25,803 in 0.0448 s) |
| Sustained streaming | **6,911 produced / 6,912 scored per sec** over 10 minutes |
| End-to-end latency at 5,000 TPS | 74–94 ms worst case |
| Live Shapley explanation | 3.26 s warm — per-decision, not batch-averaged |
| Preprocess 24,386,900 rows | 41.7 s |
| Train GNN + XGBoost | 23.0 s |
| Power-cycle to usable demo | 13 s (43 s including the 9B language model) |
| Broker loss recovery | 5 s, unattended |

### Read this before quoting an accuracy figure

The blueprint's published F1 of **0.9578 measures a dataset artefact, not detection
capability.** In the slice of IBM TabFormer that the standard benchmark tests on, **every
one of the 2,087 frauds is in the same city**. A one-line rule — `if city == "Rome"` —
scores precision 0.9803, recall 1.0000, **F1 0.9900**, and beats the trained model.

This is a property of IBM's synthetic data, whose fraud generator clusters geographically
by era: fraud spans 381 cities in 2015 and 418 in 2016, then collapses to one city from
2017. The blueprint's split (`train < 2018`, `test > 2018`) places the entire test period
inside that single-city campaign.

Re-measured on a period where fraud spans 730 cities, with every feature available:

| Test period | Fraud cities | Best single rule | Model | |
|---|---|---|---|---|
| 2019 (standard benchmark) | 1 | F1 0.9900 | 0.9578 | rule wins |
| **2015–2016** | **730** | F1 0.7568 | **0.7800** | **model wins** |

**F1 0.78 at precision 0.82 is the number to quote** — against 0.68 for the best single
rule. A smaller headline and a far stronger claim: it is the version that survives a data
science team checking it.

Details, including what this means for a real deployment, are in
[`financial-fraud-detection/docs/LIMITATIONS.md`](financial-fraud-detection/docs/LIMITATIONS.md)
and on the showcase site's *Reading the benchmark* page.

### What v2 adds

```
txn-generator ─▶ redpanda ─▶ scoring-svc ─▶ txn.scored ─▶ demo UI
                    │                                       :8090
                    ▼
              screening-stub ─▶ alert-svc ─▶ case.events ─▶ audit-svc
              (the incumbent)   (re-ranks)                  (hash-chained)
                                     │
                                     ▼
                               copilot-svc ─▶ Nemotron 9B
                               (draft narrative, EN + AR)
```

Four claims the build is designed to survive being challenged on:

- **The model cannot see the label.** Labels go to a different topic the scoring service
  does not subscribe to; the label CSV is not mounted into that container at all; and the
  service asserts this at startup and refuses to run if it ever appears.
- **The incumbent screening engine is never bypassed.** It consumes raw transactions
  directly, sees no model score, and cannot be disabled from upstream. Every alert it
  raises is preserved — the model only reorders the queue.
- **No case closes without two named people.** Enforced in the service, not the UI: a
  submitter cannot approve their own case, and calling the API directly does not help.
- **The audit log is append-only by construction.** Records arrive only from the event
  stream; the service has no write route. Altering a record breaks every hash after it,
  and verification names the offending record.

Everything is behind **one URL** (`:8090`) — live detection, screening comparison, cases,
entity graph and audit trail are tabs, not separate ports.

---

## Other demos

### Quantitative signal discovery

Based on the [NVIDIA Quantitative Signal Discovery Agent](https://github.com/NVIDIA-AI-Blueprints/quantitative-signal-discovery-agent),
adapted to serve its model locally on GB10 instead of calling NVIDIA's hosted endpoint.

| | |
|---|---|
| Full run, 3 iterations | **49–74 s** |
| Model generation | **14.7 tok/s** (30B MoE, ~3B active, FP8) |
| Backtest universe | 3,519 trading days × 380 tickers |
| Network at run time | none |

**The interesting result:** in a representative run the agents composed
`Mul(Rank(TS_Return(Close, 20)), Rank(Decay_Linear(Volume, 20)))` — momentum scaled by
volume intensity — and backtested it to a t-statistic of −5.2 over 3,494 periods. That is
statistically real, but its information coefficient was 0.012 against a 0.02 bar, so the
agent returned `best_effort` rather than claiming a win and went round the loop again. A
system that says *"significant but too weak to trade"* is behaving like a quant.

The booth screen is the Phoenix trace viewer on `:6006` — every agent call, prompt, output
and latency, live.

### Loan origination

Arabic identity and income documents in, a structured credit decision out, using a
Qwen2.5-VL-7B vision-language model with a LoRA adapter (r=64, α=128, 4 epochs, eval loss
0.139). Runs fully offline against a local 16 GB model cache.

### Video search and summarisation

Wraps the NVIDIA VSS blueprint already installed on the box rather than copying 3.4 GB into
this repo. Its UI is on `:3000`.

Note for operators: VSS bakes browser-facing URLs into container environment at **create**
time, so a network change survives restarts and shows up as websocket failures. `make vss`
re-syncs the addresses and recreates; a plain `compose start` does not. See the root
`CLAUDE.md`.

---

## Prerequisites

- NVIDIA GB10 (or any CUDA GPU with ≥32 GB) — validated on driver 595.71.05 / CUDA 13.2
- Docker 26+ with the NVIDIA Container Toolkit
- An NGC account (`docker login nvcr.io`) to pull the NVIDIA base images
- aarch64. Every image in the fraud stack has a native `linux/arm64` manifest; nothing runs
  under emulation

---

## Rebuilding from scratch

```bash
cd financial-fraud-detection
make data          # fetch IBM TabFormer (266 MB via git-LFS)
make preprocess    # build the graph from 24.4M transactions  (~42 s)
make train         # train GNN + XGBoost, install into models/ (~23 s)
make up-v2 && make prewarm
```

The 2.5 GB dataset and the Python virtualenv are deliberately **not** in this repo —
`make data` fetches the former, and the venv is machine-specific.

Preprocessing is parameterised, so the honest split above is reproducible:

```python
preprocess_data('/work/data/TabFormer', out_name='gnn_np_div',
                split_years=(2014, 2014, (2015, 2016)))
```

### Offline packaging

The demos are presented with the network cable unplugged, so everything must be on disk
first:

```bash
cd financial-fraud-detection
make offline-save     # 49 GB bundle, ~42 min
make offline-verify   # sha256 manifest, 24 s
make offline-load     # ~11.5 min
```

The bundle carries images, the language-model weights **and** the gitignored preprocessed
dataset. Provisioning takes minutes; starting takes 13 seconds.

---

## Operating on a shared box

The GB10 hosts more than these demos, and several containers elsewhere on the machine
auto-start at boot and consume nearly all of its memory. **Check what is running before
blaming a demo** — `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`, then map
the PID to its container. The launcher only knows about demos declared here.

Two traps worth knowing:

- **`docker compose up -d` does not reload bind-mounted code.** Compose recreates only when
  *configuration* changes. Use `docker compose restart <service>`.
- **Long runs degrade.** On a multi-day event, restart the stack each morning. An unattended
  18-hour run throttled the fraud pipeline from ~6,900 TPS to 26 TPS **while reporting no
  errors**. Causes and the morning checklist are in
  [`financial-fraud-detection/docs/OPERATIONS.md`](financial-fraud-detection/docs/OPERATIONS.md).

---

## The deck and the showcase

`deck/GB10-Demos-for-Events.pptx` — 27 slides covering every demo, including three on how
to read the fraud benchmark. Rebuild after changing a demo:

```bash
docker run --rm -v $PWD/deck:/deck -v $PWD/brand:/brand:ro -w /deck python:3.12-slim \
  sh -c "pip install --quiet python-pptx && python build_deck.py"
```

`showcase/` is the customer-facing site (nginx, `:8088`) — a page per use case plus
*Reading the benchmark*. Its links adapt to whatever host you browse from, so no editing is
needed when the venue network changes.

Brand assets live in `brand/` and are shared by the deck, the showcase and the demo UIs.

---

## Adding a demo

Drop it in as a subdirectory with its own `Makefile` exposing `up` and `down` (ideally also
`status`, `logs`, `verify`, `prewarm`). It is discovered automatically. Optionally:

```bash
echo mydemo > mydemo/.alias      # short name for `make mydemo`
echo myproj > mydemo/.project    # compose project name, if not the directory name
```

The `.project` file matters: the launcher stops other demos by compose project label, and a
demo whose project name it cannot infer will not be stopped when another demo starts.

---

## Notes on the data and the claims

- Dataset: [IBM TabFormer](https://github.com/IBM/TabFormer) (Apache 2.0), 24,386,900
  synthetic card transactions.
- **The benchmark's test split contains a geographic label leak.** See *Read this before
  quoting an accuracy figure* above. This is the single most important caveat in the repo.
- **Published metrics are a benchmark, not a production operating point.** Preprocessing
  undersamples the majority class, so the held-out set runs at 8.09% fraud against a true
  base rate of 0.122% (1 in 819). The demo's `realistic` mode replays at the true rate, and
  the UI shows which mode is live at all times.
- The ranking lift of the v2 comparison screen is **12–50× at the true base rate** and
  ~1.7× in demo mode, because the incumbent's queue is artificially fraud-rich when the
  fraud rate is elevated. Quote it with the mode.
- The split is temporal, so it is a backtest rather than a shuffled split.
- The UI shows ground truth on every transaction, including its false positives and misses.
  That is intentional.
- This platform performs **no sanctions or AML screening**. It prioritises and explains the
  output of a system that does.

## Attribution

`doc-element-extraction` originates from [iPulse-AI](https://github.com/iPulse-AI) and
`loan-automation` from [Efadatek](https://github.com/Efadatek), both included under
partnership agreements with Dell Technologies.

Their service credentials are development defaults, so databases, object stores and LLM
runtimes are bound to `127.0.0.1` only — reachable on the machine itself but not over the
network. Only the UI ports are exposed. **Change the credentials before any deployment
beyond a demo.**

## Trademarks

Dell Technologies and the Dell Technologies logo are trademarks of Dell Inc.
NVIDIA, the NVIDIA logo and GB10 are trademarks of NVIDIA Corporation.
Brand assets in `brand/` are included for the purpose of running these demos; white knockout
variants were generated from the supplied files (colour only — geometry unchanged). Use of
either mark is subject to the respective owner's brand guidelines.

Demo code in this repository is provided as-is for demonstration purposes.
