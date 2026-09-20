#!/usr/bin/env bash
# Is the voice agent actually ready for someone to walk up and talk to it?
set -uo pipefail
cd "$(dirname "$0")/.."
fail=0
ok(){ printf "  \033[32mPASS\033[0m  %s\n" "$1"; }
no(){ printf "  \033[31mFAIL\033[0m  %s\n" "$1"; fail=1; }
warn(){ printf "  \033[33mWARN\033[0m  %s\n" "$1"; }

echo "Nemotron Voice Agent — booth readiness"

grep -qE '^NVIDIA_API_KEY=.+' .env && ok "NVIDIA_API_KEY set" || no "NVIDIA_API_KEY missing from .env"
grep -qE '^HF_TOKEN=.+' .env && ok "HF_TOKEN set (needed for the Spark vLLM weights)" || no "HF_TOKEN missing from .env"

for i in nvcr.io/nim/nvidia/nemotron-asr-streaming:1.3.0 \
         nvcr.io/nim/nvidia/magpie-tts-multilingual:1.9.0 \
         nvcr.io/nvidia/vllm:26.05.post1-py3; do
  docker image inspect "$i" >/dev/null 2>&1 && ok "image present: ${i##*/}" || no "image missing: ${i##*/} — 'make up' will pull it"
done

code(){ curl -fsS -o /dev/null -w "%{http_code}" -m 5 "$1" 2>/dev/null; }
[ "$(code http://localhost:9001/v1/health/ready)" = 200 ] && ok "ASR ready (:9001)"  || no "ASR not ready (:9001)"
[ "$(code http://localhost:9000/v1/health/ready)" = 200 ] && ok "TTS ready (:9000)"  || no "TTS not ready (:9000)"

if curl -fsS -m 5 http://localhost:8000/v1/models >/dev/null 2>&1; then
  ok "LLM ready ($(curl -fsS -m 5 http://localhost:8000/v1/models | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null))"
else no "LLM not ready (:8000)"; fi

c=$(curl -fsSk -o /dev/null -w "%{http_code}" -m 8 https://localhost:7860/health 2>/dev/null)
[ "$c" = 200 ] && ok "app serving over HTTPS (:7860)" || no "app not serving on :7860 (got '${c:-none}')"

command -v arecord >/dev/null 2>&1 && [ -n "$(arecord -l 2>/dev/null | grep -i card)" ] \
  && ok "a capture device is visible to the host" \
  || warn "no capture device seen here — the mic belongs to the browser machine, so this is only a problem if you demo on this box"

echo
[ $fail -eq 0 ] && echo "  Ready. Open the UI, accept the cert, pick a mic, speak." \
                || echo "  Not ready — fix the failures above."
exit $fail
