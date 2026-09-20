# Patches applied to upstream model artefacts

Every modification to a file that came from the NVIDIA blueprint is recorded
here. Retraining regenerates `training/trained_models_np/` and does **not**
update the served repo at `models/prediction_and_shapley_np/`, so any patch below
must be re-applied afterwards or demo behaviour silently regresses.

`scripts/install_model.sh` re-applies P1 automatically. Run it after every
retrain.

---

## P1 — configurable Shapley sample count

**Status:** applied. **Owner:** booth demo. **Re-apply after:** any retrain.

**Files:**
- `models/prediction_and_shapley_np/1/model.py`
- `models/prediction_and_shapley_np/config.pbtxt`

**Upstream reference copy:**
`training/trained_models_np/python_backend_model_repository_np/prediction_and_shapley_np/`

**Change:**

```diff
# 1/model.py
+        self.shap_n_samples = int(parameters.get("shap_n_samples", {}).get("string_value", 32))
...
-                    n_samples=64,
+                    n_samples=self.shap_n_samples,
```

```diff
# config.pbtxt
+parameters: {
+  key: "shap_n_samples"
+  value: { string_value: "16" }
+}
```

**Why:** upstream hardcodes `n_samples=64`, giving ~12.5 s per live explanation —
too slow to hold an audience. At 16 the same explanation takes ~3.3 s.

**Floor is 16, not lower.** At `n_samples=8` the Shapley estimate is unstable
enough that feature ranking reorders between runs on the same transaction, which
is worse than slow: it makes the explanation look arbitrary. Do not reduce it to
buy latency.

**Verify after re-applying:**

```bash
grep -A3 'key: "shap_n_samples"' models/prediction_and_shapley_np/config.pbtxt
# then explain the same transaction twice and confirm the top features match
```

---

## v2 patch log

New entries go here as v2 work lands. Each needs: files touched, the diff, why,
and how to verify.

*(none yet — Stage 0 made no changes to demo code)*
