# Upstream delta — what is NVIDIA blueprint, what is local booth work

Reconstructed 2026-09-20 by inspection, because there is **no git remote
pointing at the upstream blueprint** and the machine's git history begins with
the local work. `git diff` against upstream is therefore impossible; this
document is the substitute.

Upstream project: NVIDIA AI Blueprint `financial-fraud-detection`.

## The exact local patch to upstream code

The single known local modification to an upstream file is the Shapley sample
count. The **unpatched upstream copy survives on disk**, so this diff is exact
rather than inferred:

```
upstream: training/trained_models_np/python_backend_model_repository_np/prediction_and_shapley_np/
served:   models/prediction_and_shapley_np/
```

`1/model.py` — two lines changed:

```diff
+        self.shap_n_samples = int(parameters.get("shap_n_samples", {}).get("string_value", 32))
...
-                    n_samples=64,
+                    n_samples=self.shap_n_samples,
```

`config.pbtxt` — five lines added:

```diff
+
+parameters: {
+  key: "shap_n_samples"
+  value: { string_value: "16" }
+}
```

That is the whole delta to upstream model code. Everything else local is a
**new file with no upstream equivalent**, so nothing else can be lost by an
upstream refresh.

Effect: live explanation ≈ 3.3 s at 16 samples versus ≈ 12.5 s at the upstream
default of 64. `n_samples=8` was tested and rejected — feature ranking reorders
between runs, so 16 is the floor.

## Provenance by path

### Upstream (replaceable)

| Path | Notes |
|---|---|
| `src/preprocess_TabFormer.py` | blueprint preprocessing |
| `src/preprocess_TabFormer_np.py` | node-property variant, the one in use |
| `src/preprocess_TabFormer_lp.py` | link-prediction variant, unused by the demo |
| `src/preprocess_TabFormer_np.py.bak` | local backup of the above — check before discarding |
| `docker/triton/` | blueprint Triton image build |
| `training/` | blueprint training entrypoint + config |
| `docs-upstream-README.md` | upstream README, kept for reference |

### Local — written for the booth, precious

| Path | Notes |
|---|---|
| `models/prediction_and_shapley_np/` | the **served** model repo; carries the Shapley patch above |
| `app/demo_server.py` | Flask backend, 273 lines — entirely local |
| `app/static/index.html` | booth UI, 481 lines, hand-written canvas renderer — entirely local |
| `scripts/prewarm.sh` | CUDA JIT warm-up; without it first inference costs ~85 s |
| `scripts/install_model.sh` | re-applies the Shapley patch after retraining |
| `scripts/verify.sh` | post-start health check |
| `docker/demo/`, `docker/tools/` | local image builds |
| `Makefile`, `docker-compose.yml`, `.env` | local control surface |
| `.alias`, `.project` | Money20/20 launcher integration |

`src/preprocess_TabFormer_np.py.bak` is the one genuinely ambiguous file — it is
a `.bak` (and matched by the repo-root `*.bak` ignore rule, so untracked). It
should be diffed against its live sibling before any cleanup, in case it holds a
local change that was reverted.

## Divergences from the build brief — read before coding

The brief states it was written from the runbook, not the source, and asks for
these to be reported rather than coded around.

| # | Brief says | Reality |
|---|---|---|
| 1 | `demo.sh` with `start/stop/status/restart/prewarm/urls` | **No `demo.sh` anywhere.** A `Makefile` provides all six targets plus `build clean data logs open pipeline preprocess ps rebuild train tools verify`. Parent launcher is `cd ~/APPS/Money2020 && make fraud` |
| 2 | Triton container named `tritonserver` | Named **`ffd-triton`**; demo app is **`ffd-demo`** |
| 3 | `demo_model_repo/` is the served repo | Served repo is **`models/prediction_and_shapley_np/`** |
| 4 | `trained_models_np/` at repo root | Lives at **`training/trained_models_np/`** |
| 5 | This is a repo with possible upstream remote | It is a **subdirectory** of the `Dell-Money2020` repo (10 commits, 27 tracked files here). There is no upstream remote |
| 6 | `git init` if unversioned; branch `v1-money2020` | Already versioned — but any branch operation moves **all seven demos plus the showcase site**, not just this one |
| 7 | Port 8080 out of bounds, new services 8091–8099 | Correct — but Triton also already holds **8000, 8001, 8002** on the host. Broker must avoid those too |
| 8 | Measured baseline metrics | Recorded in the runbook; **not re-measured** here. A live baseline needs the demo started, which stops VSS |

### Two items that need a decision before Stage 1

**Branching.** The brief wants `v1-money2020` kept runnable and untouched while
v2 is built on `v2-enterprise`. Because this is one repo for all demos, a branch
here freezes or forks the showcase site and six unrelated demos too. Three
options, in order of preference:

1. Branch the whole `Dell-Money2020` repo anyway and accept that the other demos
   ride along. Simplest, matches the brief, and the other demos are stable.
2. Split `financial-fraud-detection/` into its own repo with `git subtree`.
   Cleanest long-term, most disruptive now.
3. Keep one branch and isolate v2 behind a compose profile / env flag. Least
   git ceremony, but breaks acceptance criterion 1 as literally written.

**No commit has been made.** The brief's Stage 0 asks for the current state to be
committed on `v1-money2020` before anything is touched. That is explicitly on
hold at the user's instruction ("don't commit on the repo before I confirm"), so
the safety net the brief assumes is **not yet in place**. Nothing has been
modified in the demo itself — only new files added under `docs/` — so the repo
is still in its pre-v2 state and the commit can be made at any time.
