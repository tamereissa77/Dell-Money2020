# Financial Fraud Detection — GB10 booth demo (Money20/20 Middle East)
# GNN + XGBoost on NVIDIA GB10, fully self-contained: no cloud, no network at run time.

COMPOSE  := docker compose
DEMO_URL := http://localhost:$(shell grep -E '^DEMO_PORT=' .env | cut -d= -f2)
TOOLS    := ffd-tools:latest
GPU      := --gpus all
CACHE    := $(CURDIR)/cache/nvcache
DATASET  := data/TabFormer/raw/card_transaction.v1.csv

.DEFAULT_GOAL := help
.PHONY: help up down start stop restart build rebuild ps status logs clean \
        prewarm data preprocess train pipeline verify open urls tools

help:
	@echo "Demo control"
	@echo "  make up          - start Triton + demo, wait until both are healthy"
	@echo "  make down        - stop and remove containers (keeps data, models, cache)"
	@echo "  make start       - start existing containers without rebuilding"
	@echo "  make stop        - stop containers without removing them"
	@echo "  make restart     - restart both services in place"
	@echo "  make status      - health, throughput and model quality at a glance"
	@echo "  make logs        - follow logs for both services"
	@echo "  make logs-<svc>  - follow one service, e.g. make logs-triton"
	@echo "  make open        - print the booth URLs"
	@echo ""
	@echo "Booth-critical"
	@echo "  make prewarm     - warm the CUDA JIT cache (RUN AFTER EVERY COLD BOOT)"
	@echo "                     without it the first inference takes ~85s instead of 0.5s"
	@echo "  make verify      - end-to-end smoke test: score + explain a real transaction"
	@echo ""
	@echo "Pipeline (already done - only needed to rebuild from scratch)"
	@echo "  make data        - fetch the IBM TabFormer dataset (266 MB via git-LFS)"
	@echo "  make preprocess  - build the graph from 24.4M transactions (~42s)"
	@echo "  make train       - train GNN + XGBoost and emit the Triton repo (~23s)"
	@echo "  make pipeline    - data + preprocess + train, then bring the demo up"
	@echo ""
	@echo "  make build       - build all images    make rebuild - rebuild and recreate"
	@echo "  make clean       - remove containers and images (keeps data)"

# ---------- demo control ----------
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
	@curl -fsS $(DEMO_URL)/api/stats 2>/dev/null | python3 -c "\
import sys,json;d=json.load(sys.stdin);\
print(f\"  scored   : {d['n']:,} txns in {d['score_secs']}s  ({d['throughput']:,}/s)\");\
print(f\"  quality  : F1={d['f1']}  precision={d['precision']}  recall={d['recall']}\");\
print(f\"  explains : {d['hero_cached']}/{d['hero_total']} pre-computed\")" \
	|| echo "  demo not responding — try: make up"

logs:
	$(COMPOSE) logs -f

logs-%:
	$(COMPOSE) logs -f $*

open urls:
	@echo "  Booth UI : $(DEMO_URL)"
	@ip=$$(hostname -I 2>/dev/null | awk '{print $$1}'); \
	 [ -n "$$ip" ] && echo "  On LAN   : http://$$ip:$$(grep -E '^DEMO_PORT=' .env | cut -d= -f2)" || true

# ---------- booth-critical ----------
prewarm:
	@bash scripts/prewarm.sh

verify:
	@bash scripts/verify.sh

# ---------- pipeline ----------
tools:
	@docker image inspect $(TOOLS) >/dev/null 2>&1 || docker build -t $(TOOLS) ./docker/tools

data:
	@if [ -f "$(DATASET)" ]; then echo "dataset already present: $(DATASET)"; else \
	  echo "fetching IBM TabFormer (266 MB via git-LFS)..."; \
	  tmp=$$(mktemp -d); git clone --depth 1 https://github.com/IBM/TabFormer.git $$tmp/TabFormer; \
	  mkdir -p data/TabFormer/raw; \
	  cp $$tmp/TabFormer/data/credit_card/transactions.tgz data/TabFormer/raw/; \
	  tar xzf data/TabFormer/raw/transactions.tgz -C data/TabFormer/raw/; \
	  rm -rf $$tmp; echo "done: $(DATASET)"; fi

preprocess: tools
	@echo "preprocessing 24.4M transactions on GPU..."
	docker run --rm $(GPU) -v $(CURDIR):/work -v $(CACHE):/nvcache \
	  -e CUDA_CACHE_PATH=/nvcache -w /work $(TOOLS) \
	  "python -u -c \"import sys;sys.path.insert(0,'/work/src');\
from preprocess_TabFormer_np import preprocess_data;preprocess_data('/work/data/TabFormer')\""

train: tools
	@echo "training GNN + XGBoost..."
	@mkdir -p training/trained_models_np
	docker run --rm --gpus "device=0" \
	  -v $(CURDIR)/data/TabFormer/gnn_np:/data \
	  -v $(CURDIR)/training/trained_models_np:/trained_models \
	  -v $(CURDIR)/training/training_config_np.json:/app/config.json \
	  -v $(CACHE):/nvcache -e CUDA_CACHE_PATH=/nvcache \
	  nvcr.io/nvidia/cugraph/financial-fraud-training:2.0.0
	@bash scripts/install_model.sh

pipeline: data preprocess train up

clean:
	$(COMPOSE) down --rmi local
