#!/usr/bin/env bash
# Smoke test: can this demo actually run right now?
set -uo pipefail
cd "$(dirname "$0")/.."
VLLM_PORT=$(grep -E '^VLLM_PORT=' .env | cut -d= -f2)
PHX_PORT=$(grep -E '^PHOENIX_PORT=' .env | cut -d= -f2)
DATA=src/signal_discovery_workflow/data/sp500/Close.csv
fail=0
ok(){ printf "  \033[32mPASS\033[0m  %s\n" "$1"; }
no(){ printf "  \033[31mFAIL\033[0m  %s\n" "$1"; fail=1; }

echo "Quantitative signal discovery — smoke test"

[ -x .venv/bin/nat ] && ok "venv + nat CLI present" || no "venv missing — run 'make install'"
[ -f "$DATA" ] && ok "S&P 500 dataset present ($(( $(wc -l < $DATA) - 1 )) days)" || no "dataset missing — run 'make data'"
[ -d /opt/models/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 ] && ok "model weights on disk" || no "model weights missing at /opt/models"

if curl -fsS -m 10 "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
  mid=$(curl -fsS -m 10 "http://localhost:${VLLM_PORT}/v1/models" | .venv/bin/python -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
  ok "model server answering (${mid})"
  t0=$(date +%s%N)
  n=$(curl -fsS -m 180 "http://localhost:${VLLM_PORT}/v1/chat/completions" -H 'Content-Type: application/json' \
      -d '{"model":"nemotron-nano","messages":[{"role":"user","content":"Reply with the single word: ready"}],"max_tokens":8,"temperature":0}' \
      | .venv/bin/python -c "import sys,json;print(json.load(sys.stdin)['usage']['completion_tokens'])" 2>/dev/null)
  t1=$(date +%s%N)
  [ -n "$n" ] && ok "generation works (${n} tokens in $(( (t1-t0)/1000000 )) ms)" || no "generation failed"
else
  no "model server not responding on :${VLLM_PORT} — run 'make up' (allow ~2 min)"
fi

curl -fsS -m 5 "http://localhost:${PHX_PORT}" >/dev/null 2>&1 \
  && ok "Phoenix trace viewer up on :${PHX_PORT}" || no "Phoenix not responding on :${PHX_PORT}"

.venv/bin/python -c "import signal_discovery_workflow" 2>/dev/null \
  && ok "workflow package imports" || no "workflow package will not import"

echo
[ $fail -eq 0 ] && echo "  All checks passed — 'make run' is safe." || echo "  Fix the failures above before the stand opens."
exit $fail
