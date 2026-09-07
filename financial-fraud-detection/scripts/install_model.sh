#!/usr/bin/env bash
# Install a freshly trained model into the served repo, re-applying the demo patch.
#
# Training regenerates model.py from the blueprint, which hardcodes Shapley
# n_samples=64 (12.5s per explanation - too slow for a booth). This script copies
# the new artifacts into models/ and re-applies the configurable-n_samples patch,
# so retraining can never silently reintroduce the slow path.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
SRC=training/trained_models_np/python_backend_model_repository_np/prediction_and_shapley_np
DST=models/prediction_and_shapley_np
N_SAMPLES="${SHAP_N_SAMPLES:-16}"
[ -d "$SRC" ] || { echo "no trained model at $SRC - run 'make train' first"; exit 1; }

rm -rf "$DST"; mkdir -p "$DST"
cp -r "$SRC/." "$DST/"
chmod -R u+w "$DST"; rm -rf "$DST/1/__pycache__"

python3 - "$DST" "$N_SAMPLES" <<'PY'
import sys, re, pathlib
dst, n = pathlib.Path(sys.argv[1]), sys.argv[2]
mp = dst/"1"/"model.py"; s = mp.read_text()
anchor = '        self.node_type_to_predict = parameters["node_type_to_predict"]["string_value"]'
if "shap_n_samples" not in s:
    if anchor not in s: sys.exit("patch anchor missing - blueprint model.py changed upstream")
    s = s.replace(anchor, anchor +
        '\n        self.shap_n_samples = int(parameters.get("shap_n_samples", {}).get("string_value", 16))', 1)
    s2 = re.sub(r'\n(\s*)n_samples=\d+,', r'\n\1n_samples=self.shap_n_samples,', s, count=1)
    if s2 == s: sys.exit("could not rewrite n_samples call site")
    mp.write_text(s2); print("  patched model.py: n_samples is now configurable")
cp = dst/"config.pbtxt"; c = cp.read_text()
if "shap_n_samples" in c:
    c = re.sub(r'(key: "shap_n_samples"\s*\n\s*value: \{ string_value: ")\d+(" \})', r'\g<1>'+n+r'\g<2>', c)
else:
    c += '\nparameters: {\n  key: "shap_n_samples"\n  value: { string_value: "%s" }\n}\n' % n
cp.write_text(c); print(f"  shap_n_samples = {n}")
PY
echo "model installed into $DST — run 'make restart' to serve it."
