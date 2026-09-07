# Dell Technologies × NVIDIA — Money20/20 Middle East demos

Financial-services AI demos running **entirely on a single NVIDIA GB10** — no cloud,
no network at run time. Built for Money20/20 Middle East, Riyadh, 14–16 September 2026.

---

## Demos

| Demo | Alias | What it shows |
|---|---|---|
| [`financial-fraud-detection`](financial-fraud-detection/) | `fraud` | A graph neural network + XGBoost scoring 24.4M card transactions at ~589,000/sec, with a per-transaction Shapley explanation for every decision. |

### Financial fraud detection

Based on the [NVIDIA AI Blueprint for Financial Fraud Detection](https://github.com/NVIDIA-AI-Blueprints/Financial-Fraud-Detection),
adapted to run natively on GB10 (arm64) and wrapped in a booth-ready UI.

Measured on the machine, not from a datasheet:

| | |
|---|---|
| Sustained scoring | **589,528 transactions/sec** (25,803 in 0.044 s) |
| F1 / precision / recall | **0.958** · 94.3% · 97.3% |
| Preprocess 24,386,900 rows | 41.7 s |
| Train GNN + XGBoost | 23.0 s |
| Live Shapley explanation | 3.3 s |
| Power-cycle recovery | 9.4 s |

**The interesting result:** the strongest fraud signals in this dataset are *entities and
places* — merchant state (35.9%), merchant identity (34.9%), merchant city (32.5%) — not
amount or timing. A $98 clothing purchase in Rome gets flagged at 98% confidence while the
transaction amount contributes under 1% of the top driver's weight. That is precisely the
signal a row-wise model cannot see and a graph can.

---

## Running a demo

```bash
make list             # every demo, and whether it is running
make fraud            # start it
make fraud-prewarm    # warm GPU caches — REQUIRED after a cold boot
make fraud-status     # health, throughput, model quality
make fraud-verify     # end-to-end smoke test
make fraud-logs
make fraud-down
make stop             # stop everything
```

`make <demo>` also accepts the full directory name. Only one demo runs at a time — they
share a single GB10, and two model servers competing for it will run out of memory or
slow each other down.

> **After a cold boot, run `make <demo>-prewarm` first.** The Triton image ships a CUDA 12.6
> PyTorch whose kernels stop at `compute_90`; on GB10 (`sm_121`) they are JIT-compiled from
> PTX at first launch. Cold that costs ~85 s. Warm, ~0.5 s.

## Prerequisites

- NVIDIA GB10 (or any CUDA GPU with ≥32 GB) — validated on driver 595.71.05 / CUDA 13.2
- Docker 26+ with the NVIDIA Container Toolkit
- An NGC account (`docker login nvcr.io`) to pull the NVIDIA base images

## Rebuilding from scratch

```bash
cd financial-fraud-detection
make data          # fetch IBM TabFormer (266 MB via git-LFS)
make preprocess    # build the graph from 24.4M transactions  (~42 s)
make train         # train GNN + XGBoost, install into models/ (~23 s)
make up
```

The 2.5 GB dataset and the Python virtualenv are deliberately **not** in this repo —
`make data` fetches the former, and the venv is machine-specific.

## Adding a demo

Drop it in as a subdirectory with its own `Makefile` exposing `up` and `down` (ideally also
`status`, `logs`, `verify`, `prewarm`). It is discovered automatically. Optionally give it a
short name: `echo mydemo > mydemo/.alias`

---

## Notes on the data and the claims

- Dataset: [IBM TabFormer](https://github.com/IBM/TabFormer) (Apache 2.0), 24,386,900
  synthetic card transactions.
- **The published metrics are a benchmark, not a production operating point.** Preprocessing
  undersamples the majority class, so the held-out set runs at 8.09% fraud against a true
  base rate of 0.122% (1 in 819). At the real rate the false-positive economics change
  materially. Any pilot must be re-measured on live distributions.
- The training/validation/test split is temporal (before 2018 / 2018 / after 2018), so it is
  a backtest rather than a shuffled split.
- The UI shows ground truth on every transaction, including its false positives and misses.
  That is intentional.

## Trademarks

Dell Technologies and the Dell Technologies logo are trademarks of Dell Inc.
NVIDIA, the NVIDIA logo and GB10 are trademarks of NVIDIA Corporation.
Brand assets in `financial-fraud-detection/brand/` are included for the purpose of running
these demos; white knockout variants were generated from the supplied files (colour only —
geometry unchanged). Use of either mark is subject to the respective owner's brand guidelines.

Demo code in this repository is provided as-is for demonstration purposes.
