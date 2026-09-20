# Nemotron Voice Agent — spoken conversation on GB10
# ASR -> LLM -> TTS as local sidecars, WebRTC to the browser, nothing in the cloud
# at inference time.
#
# NVIDIA ships a first-class DGX Spark recipe for this blueprint, so we use theirs
# rather than adapting a workstation profile: the ASR NIM selects a dgx_spark build
# and vLLM uses the FlashInfer CUTLASS NVFP4 MoE kernel for sm_121.

COMPOSE  := docker compose
PROFILE  ?= generic-assistant/dgx-spark
APP_PORT := 7860
IP       := $(shell hostname -I 2>/dev/null | awk '{print $$1}')

.DEFAULT_GOAL := help
.PHONY: help up down start stop restart ps status logs verify prewarm open build models clean profiles

help:
	@echo "Nemotron Voice Agent — booth demo (speak to it, it speaks back)"
	@echo "  make up        - bring up the DGX Spark recipe (ASR + LLM + TTS + app)"
	@echo "  make down      - stop everything"
	@echo "  make status    - per-service health and the model each one loaded"
	@echo "  make verify    - smoke test: every sidecar answering, app serving"
	@echo "  make prewarm   - force model load so the first visitor is not the guinea pig"
	@echo "  make logs      - follow logs"
	@echo "  make open      - print the booth URL"
	@echo "  make profiles  - list the other recipes (multilingual, omni, ...)"
	@echo
	@echo "  PROFILE=omni-assistant/dgx-spark make up   - swap recipe"
	@echo
	@echo "  FIRST RUN pulls ~60-80 GB and takes 30-60 min. Do it before the stand opens."

profiles:
	@echo "  generic-assistant/dgx-spark          English cascaded ASR + LLM + TTS  (default)"
	@echo "  multilingual-assistant/dgx-spark     fixed session language, 6 LLM languages"
	@echo "  omni-assistant/dgx-spark             Omni model replaces ASR + LLM"
	@echo "  omni-assistant-subagents/dgx-spark   Omni + image/video/webcam subagents"
	@echo
	@echo "  No Arabic: no Nemotron LLM supports it (en, de, es, fr, it, ja)."

up:
	@echo "starting $(PROFILE) — first run pulls models, allow 30-60 min ..."
	$(COMPOSE) --profile $(PROFILE) up -d
	@$(MAKE) --no-print-directory open

down:
	@for p in generic-assistant/dgx-spark multilingual-assistant/dgx-spark \
	          omni-assistant/dgx-spark omni-assistant-subagents/dgx-spark; do \
	  $(COMPOSE) --profile $$p down 2>/dev/null; done; true

start:
	$(COMPOSE) --profile $(PROFILE) start
stop:
	$(COMPOSE) --profile $(PROFILE) stop
restart:
	$(COMPOSE) --profile $(PROFILE) restart
build:
	$(COMPOSE) --profile $(PROFILE) build

ps status:
	@$(COMPOSE) --profile $(PROFILE) ps
	@echo
	@printf "  app (7860)  : "; curl -fsSk -o /dev/null -w "HTTP %{http_code}\n" -m 5 https://localhost:$(APP_PORT)/health 2>/dev/null || echo "not responding"
	@printf "  ASR  (9001) : "; curl -fsS -o /dev/null -w "HTTP %{http_code}\n" -m 5 http://localhost:9001/v1/health/ready 2>/dev/null || echo "not ready"
	@printf "  TTS  (9000) : "; curl -fsS -o /dev/null -w "HTTP %{http_code}\n" -m 5 http://localhost:9000/v1/health/ready 2>/dev/null || echo "not ready"
	@printf "  LLM         : "; curl -fsS -m 5 http://localhost:8000/v1/models 2>/dev/null \
	  | python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "not ready"

logs:
	$(COMPOSE) --profile $(PROFILE) logs -f

# The sidecars load tens of GB of weights. Until they have, the first person to
# speak waits through it. Run this once after `make up` reports healthy.
prewarm:
	@echo "waiting for every sidecar to report ready (can take several minutes) ..."
	@for i in $$(seq 1 120); do \
	  a=$$(curl -fsS -o /dev/null -w "%{http_code}" -m 5 http://localhost:9001/v1/health/ready 2>/dev/null); \
	  t=$$(curl -fsS -o /dev/null -w "%{http_code}" -m 5 http://localhost:9000/v1/health/ready 2>/dev/null); \
	  l=$$(curl -fsS -o /dev/null -w "%{http_code}" -m 5 http://localhost:8000/v1/models 2>/dev/null); \
	  if [ "$$a" = 200 ] && [ "$$t" = 200 ] && [ "$$l" = 200 ]; then echo "  all sidecars ready"; break; fi; \
	  printf "\r  asr=%s tts=%s llm=%s  (%ss)" "$$a" "$$t" "$$l" "$$((i*10))"; sleep 10; done; echo
	@echo "now open the UI and speak one sentence — that warms the audio path end to end."

verify:
	@bash scripts/booth-verify.sh

open:
	@echo "  Booth UI : https://localhost:$(APP_PORT)     (self-signed cert — accept the warning)"
	@[ -n "$(IP)" ] && echo "  On LAN   : https://$(IP):$(APP_PORT)" || true
	@echo "  Use a wired headset. Built-in laptop mics pick up the stand and break turn-taking."

clean:
	@$(MAKE) --no-print-directory down
	$(COMPOSE) down --remove-orphans 2>/dev/null; true
