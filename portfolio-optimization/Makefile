# Portfolio optimization — NVIDIA cuOpt + RAPIDS cuML on GB10
# Mean-CVaR optimisation and rebalancing backtests, GPU vs CPU, fully on-device.

COMPOSE := docker compose
PORT    := $(shell grep -E '^APP_PORT=' .env | cut -d= -f2)
URL     := http://localhost:$(PORT)
DATASET := data/stock_data/sp500.csv

.DEFAULT_GOAL := help
.PHONY: help up down start stop restart build rebuild ps status logs clean data bench verify open

help:
	@echo "Portfolio optimization (cuOpt) — booth demo"
	@echo "  make up        - start the Streamlit demo, wait until healthy"
	@echo "  make down      - stop and remove the container"
	@echo "  make status    - container health + dataset presence"
	@echo "  make logs      - follow logs"
	@echo "  make data      - download S&P 500 price history (one-time, needs internet)"
	@echo "  make bench     - GPU vs CPU Mean-CVaR benchmark (the headline number)"
	@echo "  make verify    - smoke test: app healthy, dataset present, cuOpt importable"
	@echo "  make open      - print the booth URL"
	@echo "  make build / rebuild / clean"

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
build:
	$(COMPOSE) build
rebuild:
	$(COMPOSE) up -d --build --force-recreate --wait

ps status:
	@$(COMPOSE) ps
	@if [ -f "$(DATASET)" ]; then \
	  echo "  dataset : $$(wc -l < $(DATASET)) rows, $$(head -1 $(DATASET) | tr ',' '\n' | wc -l) columns"; \
	else echo "  dataset : MISSING — run 'make data'"; fi
	@curl -fsS -o /dev/null -w "  app     : HTTP %{http_code}\n" $(URL)/_stcore/health 2>/dev/null \
	  || echo "  app     : not responding — try 'make up'"

logs:
	$(COMPOSE) logs -f

open:
	@echo "  Booth UI : $(URL)"
	@ip=$$(hostname -I 2>/dev/null | awk '{print $$1}'); \
	 [ -n "$$ip" ] && echo "  On LAN   : http://$$ip:$(PORT)" || true

# one-time: pulls S&P 500 closing prices 2005-2025 from Yahoo Finance.
# After this the demo runs entirely offline.
data: build
	@if [ -f "$(DATASET)" ]; then echo "dataset already present: $(DATASET)"; else \
	  echo "downloading S&P 500 price history (needs internet, ~1 min) ..."; \
	  $(COMPOSE) run --rm --no-deps -T portfolio \
	    python -c "from portfolio_optimization.utils import download_data; download_data('data/stock_data', datasets=['sp500'])"; \
	fi
	@ls -la data/stock_data/ 2>/dev/null | tail -3

bench: build
	@echo "Mean-CVaR GPU (cuOpt) vs CPU (CVXPY) — 500 assets x 50,000 scenarios"
	@$(COMPOSE) run --rm --no-deps -T portfolio python /work/demo/gb10_bench.py 500 50000

verify:
	@bash scripts/verify.sh

clean:
	$(COMPOSE) down --rmi local
