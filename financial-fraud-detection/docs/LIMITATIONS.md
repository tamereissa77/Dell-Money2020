# Limitations — read before demonstrating this to a bank

## 1. The test set contains a label leak, and it dominates the headline metrics

**Every one of the 2,087 frauds in the preprocessed test set is in the city
"Rome".** There are no frauds anywhere else.

```
rows in Rome        : 2,129
  ...fraudulent     : 2,087
  ...clean          :    42
frauds outside Rome :     0

P(fraud | Rome)     : 0.9803
P(fraud | not Rome) : 0.0000
```

A one-line rule — `if city == "Rome": fraud` — therefore scores:

| | Precision | Recall | F1 |
|---|---|---|---|
| `city == "Rome"` | 0.9803 | **1.0000** | **0.9900** |
| GraphSAGE + XGBoost | 0.9429 | 0.9732 | 0.9578 |

**The trivial rule beats the model.** The published F1 of 0.9578 is not evidence
of fraud-detection capability on this test set; it is a model recovering a
dataset artefact, slightly imperfectly.

This also explains the demo's signature explanation. The v1 booth demo's
headline Shapley output is *"top driver: Merchant city +10.43"* — the model is
telling us exactly this, and we were reading it as insight.

### It is not in the source data

The raw TabFormer file has fraud spread across **366 distinct cities**:

```
ONLINE 1,950 | Rome 616 | Algiers 94 | Port au Prince 40 | Strasburg 37 | ...
```

So the leak is introduced by the preprocessing in `src/preprocess_TabFormer_np.py`,
not by IBM's data. The test set spans 2019-01-01 to 2020-12-31 and contains
4,617 distinct cities, so it is **not** a narrow temporal slice that happens to
be Rome-heavy — the frauds specifically collapsed to one city. The exact
mechanism is not yet identified; the under-sampling at
`preprocess_TabFormer_np.py:316` keeps all frauds and samples the majority
class, so the filtering happens earlier.

### What to do about it

Until this is resolved, do **not** present F1, precision or recall from this
test set as evidence of detection capability. What the demo can still honestly
show:

- the engineering: streaming ingestion, GPU scoring throughput, per-decision
  Shapley attribution, the audit trail
- the **ranking** argument in `/compare`, which measures ordering within the
  incumbent's queue rather than absolute detection quality

What it cannot show: that the model detects fraud better than a rule, on this
data.

Fixing it means re-running preprocessing so that the test-set fraud retains the
source distribution, then retraining and re-measuring everything. Expect the
honest F1 to be materially lower.

### Consequence for acceptance criterion 8

Criterion 8 asks for a typology the model catches and the incumbent's screening
misses. **It cannot be satisfied on this data.** Any rule set with Rome on its
watchlist catches 100% of frauds, and one without it catches none of them by
geography. Zero of the 22,787 rule-evading rows are fraudulent.

The `mule-fanin-fanout` builder was written to find fraudulent rows that evade
the incumbent's rules; it correctly reports `unavailable` rather than
fabricating a result.

---

## 2. Benchmark, not a production operating point

The test set is rebalanced to **8.09% fraud**. The true TabFormer base rate is
**0.122%** — a factor of 66. Precision figures computed on the rebalanced set do
not transfer to production traffic.

The generator's `realistic` mode replays at the true base rate, and the
comparison screen shows the active mode alongside every number.

## 3. The incumbent stub is a stand-in, and its watchlist is illustrative

`screening-stub` is not a certified screening engine. Its watchlist is a
plausible shape, not a real sanctions list. Its measured false-positive rate
(98–99% at the true base rate) is in the band real engines occupy, but its
recall is an artefact of limitation 1 above.

## 4. The ranking lift is a range, not a constant

Measured lift at top-500 has ranged from **11.8x to 47x** across runs on the
same pipeline, because it depends on how many frauds happen to be in the
current queue window. Quote it as a range and say what window it came from.

## 5. Scope and hardware

- Public dataset, US card data, 2019–2020.
- Synthetic typologies assembled from real rows, not observed criminal cases.
- Desktop-class hardware (one GB10), single node, no HA.
- No production data lineage: `data_version` identifies a file, not a governed
  pipeline.
- The platform performs **no sanctions or AML screening**. It prioritises and
  explains the output of a system that does.

## 6. Long-run stability

Triton's inference latency degrades from ~27 ms to several seconds over a
multi-day run and must be restarted daily. See `OPERATIONS.md`.
