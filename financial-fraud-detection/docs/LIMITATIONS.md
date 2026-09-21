# Limitations — read before demonstrating this to a bank

## 1. The published F1 of 0.9578 measures a dataset artefact

**Every one of the 2,087 frauds in the blueprint's test split is in the city
"Rome".** There are no frauds anywhere else.

```
rows in Rome        : 2,129
  ...fraudulent     : 2,087
  ...clean          :    42
frauds outside Rome :     0
```

A one-line rule — `if city == "Rome": fraud` — scores precision 0.9803,
recall 1.0000, **F1 0.9900**, beating the trained model's 0.9578. This also
explains the demo's signature Shapley output, *"top driver: Merchant city"*:
the model learned the shortcut and was telling us so.

### Where it comes from

**This is a property of IBM's TabFormer data, not of our preprocessing.** Its
synthetic fraud generator clusters fraud geographically by era:

| Year | Frauds | Distinct cities | Top |
|---|---|---|---|
| 2015 | 3,281 | 381 | ONLINE 2,777 |
| 2016 | 3,579 | 418 | ONLINE 3,073 |
| 2017 | 255 | **1** | Rome 255 |
| 2018 | 2,491 | 4 | Rome 2,340 |
| 2019 | 2,087 | **1** | Rome 2,087 |

The blueprint splits `train < 2018`, `val = 2018`, `test > 2018` — which places
the **entire test period inside the Rome era**. In 2019 all frauds are also in
State "Italy" with a blank Zip, so City, State and Zip all leak identically.

### Measured three ways

| Setup | Test period | Fraud cities | F1 | Precision | Recall |
|---|---|---|---|---|---|
| All features (**the shipped demo**) | 2019 | 1 | **0.9578** | 0.9429 | 0.9732 |
| City + Zip removed | 2019 | 1 | **0.5304** | 0.6344 | 0.4557 |
| All features | **2015–2016** | **730** | **0.7800** | 0.8212 | 0.7427 |

Read these together:

- Dropping two columns collapses the shipped configuration from 0.96 to 0.53.
  That is how much of the headline was geography.
- But the model is **not** merely a Rome-detector. Evaluated on a period where
  fraud spans 730 cities, with every feature available, it reaches **F1 0.78**
  — a real, defensible result.

**F1 0.78 is the number to quote.** It comes from
`gnn_np_div` (split `train < 2014`, `val = 2014`, `test = 2015–2016`) and
`training/trained_models_div`.

Reproduce with:

```python
preprocess_data('/work/data/TabFormer', out_name='gnn_np_div',
                split_years=(2014, 2014, (2015, 2016)))
```

### Two incidental findings

The 2015–16 test set is dominated by ONLINE fraud (5,850 of 6,860). That is a
legitimate signal — card-not-present fraud genuinely concentrates online — but
it should be stated rather than presented as subtle graph reasoning.

Building that split also exposed a real temporal-shift problem: `OneHotEncoder`
raised `Found unknown categories ['Chip Transaction']`, because chip cards did
not exist before ~2014. It now uses `handle_unknown="ignore"`, which encodes an
unseen category as all-zeros — the correct behaviour for a temporal split, and
a reminder that a model trained on one payment era cannot see a later one.

### Consequence for acceptance criterion 8

Criterion 8 asks for a typology the model catches and the incumbent's screening
misses. **It cannot be satisfied on the 2019 split**: any rule set with Rome on
its watchlist catches 100% of frauds, and zero of the 22,787 rule-evading rows
are fraudulent. The `mule-fanin-fanout` builder reports `unavailable` rather
than fabricating a result. On the 2015–2016 split it becomes answerable, and
that is where it should be demonstrated.

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
