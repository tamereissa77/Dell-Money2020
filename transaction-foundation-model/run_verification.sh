#!/usr/bin/env bash
# Unattended verification of the transaction-foundation-model claim:
#   do foundation-model embeddings actually improve fraud detection?
# Runs notebooks 01 -> 02 -> 04 -> 05 (03 is pretraining; we use the shipped
# checkpoint instead). Writes progress and a summary to results/.
set -uo pipefail
cd /workspace
R=results; mkdir -p "$R"
say(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$R/progress.log"; }

say "START — transaction-foundation-model verification on GB10"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | sed 's/^/  gpu: /' | tee -a "$R/progress.log"

run_nb(){
  local nb=$1 label=$2 t0 rc
  say "RUN  $nb  ($label)"
  t0=$(date +%s)
  timeout 21600 jupyter nbconvert --to notebook --execute "$nb" \
      --output "$R/executed_$nb" --ExecutePreprocessor.timeout=20000 \
      >> "$R/$nb.log" 2>&1
  rc=$?
  local dt=$(( $(date +%s) - t0 ))
  if [ $rc -eq 0 ]; then say "DONE $nb in ${dt}s"; else say "FAIL $nb rc=$rc after ${dt}s (see $R/$nb.log)"; fi
  return $rc
}

run_nb 01_dataset_baseline.ipynb            "TabFormer load + XGBoost baseline"
run_nb 02_seq_preproc_tokenization.ipynb    "GPU tokenizer -> decoder corpus"
run_nb 04_inference_embedding_extraction.ipynb "512-dim embeddings from shipped checkpoint"
run_nb 05_xgboost_fraud_detection.ipynb     "raw vs embeddings vs combined"

say "extracting headline numbers ..."
python - <<'PY' | tee -a "$R/summary.txt"
import json, glob, re, os
print("\n=== VERDICT: do foundation-model embeddings improve fraud detection? ===\n")
hits=[]
for f in sorted(glob.glob("results/executed_0*.ipynb")):
    try: nb=json.load(open(f))
    except Exception: continue
    for c in nb.get("cells",[]):
        for o in c.get("outputs",[]):
            txt="".join(o.get("text","")) if o.get("output_type")=="stream" else \
                "".join(o.get("data",{}).get("text/plain",""))
            for line in txt.splitlines():
                if re.search(r"(AUPRC|average.?precision|AUC|raw|embedding|combined|F1|recall|precision)", line, re.I) \
                   and re.search(r"\d\.\d{2,}", line):
                    hits.append(f"  [{os.path.basename(f)[9:11]}] {line.strip()[:150]}")
seen=set(); out=[h for h in hits if not (h in seen or seen.add(h))]
print("\n".join(out[-60:]) if out else "  (no metric lines captured — inspect results/executed_05*.ipynb)")
PY
say "COMPLETE — see results/summary.txt and results/progress.log"
