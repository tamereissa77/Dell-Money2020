# Quantitative signal discovery — NVIDIA NeMo Agent Toolkit on GB10
# Three agents in a loop: invent an alpha signal, write it as Python, backtest it.
# Model served locally; after `make data` nothing touches the network.

COMPOSE  := docker compose
VLLM_PORT   := $(shell grep -E '^VLLM_PORT=' .env | cut -d= -f2)
PHX_PORT    := $(shell grep -E '^PHOENIX_PORT=' .env | cut -d= -f2)
MODEL_DIR   := $(shell grep -E '^MODEL_HOST_DIR=' .env | cut -d= -f2)
PY       := .venv/bin/python
NAT      := .venv/bin/nat
CFG      := configs/config-gb10-local.yml
DATA     := src/signal_discovery_workflow/data/sp500/Close.csv
OUT      := src/signal_discovery_workflow/output
INPUT    ?= momentum signals

.DEFAULT_GOAL := help
.PHONY: help up down start stop restart ps status logs verify prewarm run run-% data install open clean results

help:
	@echo "Quantitative signal discovery (NeMo Agent Toolkit) — booth demo"
	@echo "  make up        - start the local model server + Phoenix trace viewer"
	@echo "  make down      - stop both"
	@echo "  make run       - discover signals (default: \"$(INPUT)\")"
	@echo "  make run-vol   - volatility signals   (also: run-rev, run-volume)"
	@echo "  make results   - show the latest discovered signal and its metrics"
	@echo "  make status    - services, model, dataset, last result"
	@echo "  make verify    - smoke test: model answers, data present, workflow imports"
	@echo "  make prewarm   - load the model into GPU memory (REQUIRED after cold boot)"
	@echo "  make data      - download S&P 500 history (one-time, needs internet)"
	@echo "  make install   - build the Python venv from the lockfile"
	@echo "  make open      - print the booth URL"
	@echo
	@echo "  INPUT=\"mean reversion signals\" make run   - any prompt you like"

up:
	$(COMPOSE) up -d --wait
	@$(MAKE) --no-print-directory open

down:
	$(COMPOSE) down

start:
	$(COMPOSE) start
stop:
	$(COMPOSE) stop
restart:
	$(COMPOSE) restart

ps status:
	@$(COMPOSE) ps
	@printf "  model   : "; curl -fsS -m 5 http://localhost:$(VLLM_PORT)/v1/models 2>/dev/null \
	  | $(PY) -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'],'ready')" 2>/dev/null \
	  || echo "not responding — 'make up', then allow ~2 min to load"
	@printf "  phoenix : "; curl -fsS -o /dev/null -w "HTTP %{http_code}\n" http://localhost:$(PHX_PORT) 2>/dev/null || echo "not responding"
	@if [ -f "$(DATA)" ]; then \
	  printf "  dataset : %s trading days x %s tickers\n" \
	    "$$(($$(wc -l < $(DATA)) - 1))" "$$(($$(head -1 $(DATA) | tr ',' '\n' | wc -l) - 1))"; \
	else echo "  dataset : MISSING — run 'make data'"; fi
	@printf "  results : "; ls -1 $(OUT)/*.json 2>/dev/null | wc -l | tr -d '\n'; echo " saved run(s)"

logs:
	$(COMPOSE) logs -f

# The 30B MoE loads ~31 GB of weights from disk. Cold that is ~2 min; warm it is instant.
# Run this once after a power cycle, before anyone is watching.
prewarm: up
	@echo "warming the model (first token forces weight load + JIT) ..."
	@time curl -fsS -m 900 http://localhost:$(VLLM_PORT)/v1/chat/completions \
	  -H 'Content-Type: application/json' \
	  -d '{"model":"nemotron-nano","messages":[{"role":"user","content":"ok"}],"max_tokens":1}' \
	  >/dev/null && echo "  model warm"

run:
	@echo "discovering: \"$(INPUT)\"  (typically ~50 s for 3 iterations)"
	@$(NAT) run --config_file $(CFG) --input "$(INPUT)"
	@$(MAKE) --no-print-directory results

run-vol:
	@$(MAKE) --no-print-directory run INPUT="volatility signals"
run-rev:
	@$(MAKE) --no-print-directory run INPUT="mean reversion signals"
run-volume:
	@$(MAKE) --no-print-directory run INPUT="volume price divergence signals"

results:
	@$(PY) scripts/show_result.py

verify:
	@bash scripts/verify.sh

data:
	@if [ -f "$(DATA)" ]; then echo "dataset already present"; else \
	  echo "downloading S&P 500 history from Yahoo Finance (needs internet, ~30 s) ..."; \
	  $(PY) -m signal_discovery_workflow.download_data; fi
	@ls -la src/signal_discovery_workflow/data/sp500/ 2>/dev/null | tail -6

install:
	@command -v uv >/dev/null 2>&1 || { python3 -m venv .uvbootstrap && .uvbootstrap/bin/pip install -q uv; }
	@PATH="$$PWD/.uvbootstrap/bin:$$PATH" uv sync --locked
	@echo "venv ready: $(PY)"

open:
	@echo "  Traces  : http://localhost:$(PHX_PORT)    <- the booth screen"
	@echo "  Model   : http://localhost:$(VLLM_PORT)/v1/models"
	@ip=$$(hostname -I 2>/dev/null | awk '{print $$1}'); \
	 [ -n "$$ip" ] && echo "  On LAN  : http://$$ip:$(PHX_PORT)" || true

clean:
	$(COMPOSE) down
	@rm -rf $(OUT)/*.json traces/*
