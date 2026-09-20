# Baseline — captured before any v2 work

Captured 2026-09-20 on `gb10-galeneai`.

## Absolute path and account

```
/home/delluser26/APPS/Money2020/financial-fraud-detection
user: delluser26   home: /home/delluser26
```

The runbook's `/home/delluser26` is **still correct** — that is the live account.

## Git

This directory is **not its own repository**. It is a subdirectory of:

```
toplevel: /home/delluser26/APPS/Money2020
remote:   https://github.com/tamereissa77/Dell-Money2020.git
branch:   main    commits: 10    tracked files under this demo: 27
```

There is **no remote pointing at the upstream NVIDIA blueprint**, so `git diff`
against upstream is not possible. Provenance was reconstructed by inspection
instead — see `UPSTREAM_DELTA.md`.

Consequence for the brief: `git checkout v1-money2020` would switch the **entire
Money20/20 repo** (all seven demos plus the showcase site), not just this demo.
See the divergence note in `UPSTREAM_DELTA.md`.

## Control surface

There is **no `demo.sh`**, at this level or in `~/APPS/Money2020/`. Control is a
`Makefile` here, plus the parent launcher (`cd ~/APPS/Money2020 && make fraud`).

Targets present: `build clean data down help logs open pipeline preprocess
prewarm ps rebuild restart start status stop tools train up urls verify`

Every subcommand the brief names on `demo.sh` exists as a make target except
`offline-save`, `offline-load`, `verify-audit` and `export-case`, which are new
in v2.

## `make status` output, verbatim

Captured with the demo **not running**:

```
NAME      IMAGE     COMMAND   SERVICE   CREATED   STATUS    PORTS
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/usr/lib/python3.12/json/__init__.py", line 293, in load
    return loads(fp.read(),
           ^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/json/__init__.py", line 346, in loads
    return _default_decoder.decode(s)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/json/decoder.py", line 337, in decode
    obj, end = self.raw_decode(s, idx=_w(s, 0).end())
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/json/decoder.py", line 355, in raw_decode
    raise JSONDecodeError("Expecting value", s, err.value) from None
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
  demo not responding — try: make up
```

**This is a live defect.** `status` pipes an empty/non-JSON response into
`json.load` without guarding, so the not-running path emits a Python traceback
before the intended message. It is cosmetic but it is the first thing a customer
sees if the demo is down. Fix is in scope for Stage 0 and does not change the
subcommand's contract (same exit behaviour, same final line).

### Live capture (VSS stopped by the user, box otherwise idle)

```
NAME         IMAGE               SERVICE   STATUS                    PORTS
ffd-demo     ffd-demo:latest     demo      Up 13 seconds (healthy)   0.0.0.0:8090->8090/tcp
ffd-triton   ffd-triton:latest   triton    Up 24 seconds (healthy)   0.0.0.0:8000-8002->8000-8002/tcp
  scored   : 25,803 txns in 0.0448s  (575,775/s)
  quality  : F1=0.9578  precision=0.9429  recall=0.9732
  explains : 3/12 pre-computed
```

`make verify`: **ALL CHECKS PASSED** — triton health 200, demo api 200, scoring
OK, feed has data, live explanation top driver `Merchant city` +10.43.

The `status` traceback above does not appear when the stack is up; it is
specific to the not-running path.

## Machine state at capture

Initial capture, with VSS running and the fraud demo down:

```
RAM : 70 GB used / 121 total — 51 GB available
GPU : 58,527 MiB in use   containers: 11 (all VSS / project mdx)
```

After the user stopped VSS, **with the fraud demo running**:

```
RAM : 7 GB used / 121 total — 114 GB available
GPU : 592 MiB total (237 MiB + 355 MiB, both ffd-triton)
DISK: /  1.9T used, 1.6T avail (55%)
```

**This is the headroom answer for v2 (brief §11 Q1).** The running v1 demo costs
about **3 GB of RAM and under 600 MiB of GPU** — far less than assumed. Roughly
114 GB RAM and effectively the whole GPU remain free. The full v2 stack fits
comfortably, including a local LLM for `copilot-svc` alongside Triton. Nothing
needs to be cut for capacity reasons.

## Images

| Image | Size | Arch |
|---|---|---|
| `ffd-triton:latest` | 36.3 GB | linux/arm64 |
| `ffd-demo:latest` | 25.5 GB | linux/arm64 |
| `ffd-preprocess:latest` | 25.6 GB | — |
| `ffd-tools:latest` | 25.6 GB | — |

Both runtime images are **natively arm64** — no emulation, satisfying constraint 3.

Docker overall: 350 images / 876 GB, 225 GB build cache, 471 GB reclaimable.

## Services and ports

From `docker-compose.yml` + `.env`:

| Service | Container | Host port |
|---|---|---|
| `triton` | `ffd-triton` | 8000 (HTTP), 8001 (gRPC), 8002 (metrics) |
| `demo` | `ffd-demo` | **8090** |

Container names are `ffd-triton` / `ffd-demo` — **not** `tritonserver` as the
brief states. Port 8090 confirmed; 8080 is untouched and belongs to
`openshell-gateway`.

Note for v2: Triton already occupies host ports **8000–8002**. The brief
allocates new services in 8091–8099, which does not collide.

## Model repository

- **Served (patched):** `models/prediction_and_shapley_np/` — *not*
  `demo_model_repo/` as the brief states.
- **Training output:** `training/trained_models_np/` — *not* `trained_models_np/`
  at the repo root.
- The unpatched upstream copy survives at
  `training/trained_models_np/python_backend_model_repository_np/`, which is what
  made an exact upstream diff possible.

`shap_n_samples` is set to **16** in the served `config.pbtxt`.

## Data

| Path | Size |
|---|---|
| `data/TabFormer/raw/card_transaction.v1.csv` | 2.2 GB |
| `data/TabFormer/raw/transactions.tgz` | 266 MB |
| `data/TabFormer/gnn_np/` (preprocessed GNN artefacts) | 43 MB |
| `data/TabFormer` total | 2.5 GB |

Data lives **inside** the repo but is gitignored (`*/data/` at the repo root),
so it is on disk but not version-controlled. The demo server reads
`data/TabFormer/gnn_np/test_gnn` via `GNN_TEST_DIR`.

Offline packaging (brief §3.2) must therefore account for 2.5 GB of data that is
not in git **and** ~62 GB of images.

## Measured latency

| Measurement | Value |
|---|---|
| Batch scoring | 25,803 txns in 0.0448 s = **575,775/s** (recorded: 589,528/s — within 2.3%) |
| Explanation, **cold** (before `make prewarm`) | **5.31 s** |
| Explanation, **warm** (after `make prewarm`) | **3.256 s**, repeatable |
| Pre-computed hero explanation | 0.0015 s (served from cache) |
| `make prewarm` duration | 15.1 s |

The cold/warm gap matters for acceptance criterion 6 (≤ 3.5 s): the demo
**fails it cold and passes it warm**. `prewarm` is not optional.

## Demo application

`app/demo_server.py` — 273 lines, Flask. `app/static/index.html` — 481 lines,
single page, hand-written canvas renderer, no CDN.

Endpoints:

```
GET /                     GET /api/stats      GET /api/feed
GET /api/heroes           GET /api/brand      GET /api/explain/<int:i>
GET /brand/<path:fn>      @app.after_request  (no-cache headers)
```

It talks to Triton over `tritonclient.http` with a thread-local client, and has a
SIGTERM handler to avoid exit 137.
