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
	@$(COMPOSE) up -d redpanda-init >/dev/null 2>&1 || true
	$(COMPOSE) up -d --wait $(V2_SERVICES)
	@$(MAKE) --no-print-directory open

# --- v1 / v2 variants ------------------------------------------------------
# v1 is the original booth demo: Triton plus the UI reading the pre-scored
# file. Nothing else. On a shared GB10 that matters - v2 adds six containers
# and a 20 GB language model, and a v1 audience should not be paying for them.
#
# V1_SERVICES is the whole v1 surface; everything else is torn down rather than
# left idle, because "stopped but present" still holds GPU on this box.
# `compose up --wait` treats a one-shot that exits 0 as a failure, so
# redpanda-init (which creates the topics and is *meant* to exit) has to be
# excluded from any --wait or the whole start reports an error while actually
# having succeeded. Listing the long-running services is the fix.
V1_SERVICES := triton demo
V2_SERVICES := redpanda triton demo txn-generator scoring-svc screening-stub \
               alert-svc audit-svc copilot-svc copilot-llm
V2_ONLY     := redpanda txn-generator scoring-svc screening-stub alert-svc \
               audit-svc copilot-svc copilot-llm

up-v1:
	@echo "  v1 booth demo: Triton + UI on the pre-scored file"
	@$(COMPOSE) rm -sf $(V2_ONLY) >/dev/null 2>&1 || true
	@FEED_SOURCE=file $(COMPOSE) up -d --wait $(V1_SERVICES)
	@$(MAKE) --no-print-directory open

up-v2:
	@echo "  v2 enterprise pipeline: streaming, screening, cases, audit, copilot"
	@FEED_SOURCE=kafka $(COMPOSE) up -d redpanda-init >/dev/null 2>&1 || true
	@FEED_SOURCE=kafka $(COMPOSE) up -d --wait $(V2_SERVICES)
	@$(MAKE) --no-print-directory open

which:
	@running=$$(docker compose ps --services --status running 2>/dev/null | tr '\n' ' '); \
	  case "$$running" in \
	    *redpanda*) echo "  variant: v2 (full pipeline)";; \
	    *demo*)     echo "  variant: v1 (booth demo)";; \
	    *)          echo "  variant: not running";; \
	  esac
	@$(MAKE) --no-print-directory mode 2>/dev/null | head -1

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

# --- offline bundle --------------------------------------------------------
# The demo is presented with the network cable unplugged. Everything it needs
# has to be on disk first: images, the Nemotron weights (a named volume, not an
# image layer - see the comment in docker-compose.yml) and the preprocessed
# TabFormer data, which is gitignored and therefore not in a clone.
OFFLINE ?= offline
.PHONY: offline-save offline-load offline-verify

offline-save:
	@mkdir -p $(OFFLINE)/images
	@echo "saving images (this is tens of GB - expect minutes, not seconds)"
	@imgs=$$(docker compose config 2>/dev/null | python3 -c "import sys,yaml;d=yaml.safe_load(sys.stdin);print(' '.join(sorted({s['image'] for s in d['services'].values() if s.get('image')})))"); \
	  echo "  $$imgs"; \
	  start=$$(date +%s); \
	  docker save $$imgs | gzip -1 > $(OFFLINE)/images/stack.tar.gz; \
	  echo "  images: $$(du -h $(OFFLINE)/images/stack.tar.gz | cut -f1) in $$(( $$(date +%s) - start ))s"
	@echo "saving the model weights volume"
	@docker run --rm -v financial-fraud-detection_nemotron-hf:/src -v $(CURDIR)/$(OFFLINE):/out \
	  alpine tar czf /out/nemotron-hf.tar.gz -C /src . 2>/dev/null
	@echo "  weights: $$(du -h $(OFFLINE)/nemotron-hf.tar.gz | cut -f1)"
	@echo "saving the preprocessed dataset"
	@tar czf $(OFFLINE)/tabformer-gnn.tar.gz data/TabFormer/gnn_np 2>/dev/null
	@echo "  data: $$(du -h $(OFFLINE)/tabformer-gnn.tar.gz | cut -f1)"
	@cd $(OFFLINE) && sha256sum images/stack.tar.gz nemotron-hf.tar.gz tabformer-gnn.tar.gz > MANIFEST.sha256
	@echo "manifest:"; sed 's|^|  |' $(OFFLINE)/MANIFEST.sha256
	@echo "total: $$(du -sh $(OFFLINE) | cut -f1)"

offline-verify:
	@cd $(OFFLINE) && sha256sum -c MANIFEST.sha256 | sed 's|^|  |'

offline-load:
	@test -f $(OFFLINE)/MANIFEST.sha256 || { echo "no bundle in $(OFFLINE)/" >&2; exit 1; }
	@$(MAKE) --no-print-directory offline-verify
	@echo "loading images ..."; start=$$(date +%s); \
	  gunzip -c $(OFFLINE)/images/stack.tar.gz | docker load | tail -2; \
	  echo "  loaded in $$(( $$(date +%s) - start ))s"
	@docker volume create financial-fraud-detection_nemotron-hf >/dev/null
	@docker run --rm -v financial-fraud-detection_nemotron-hf:/dest -v $(CURDIR)/$(OFFLINE):/in \
	  alpine tar xzf /in/nemotron-hf.tar.gz -C /dest
	@tar xzf $(OFFLINE)/tabformer-gnn.tar.gz
	@echo "loaded. now run: make up && make prewarm"

# --- v2 feed mode ----------------------------------------------------------
# The UI reads either the pre-scored file (v1 behaviour, the default) or the
# live txn.scored stream. It is a container env var fixed at create time, so it
# needs a recreate, not a restart - and a plain `docker compose up -d demo`
# silently reverts to the .env default. These two targets make the mode an
# explicit choice instead of something you have to remember.
.PHONY: stream classic mode up-v1 up-v2 which

stream:
	@FEED_SOURCE=kafka $(COMPOSE) up -d demo
	@printf "  waiting for the UI "; for i in $$(seq 1 40); do \
	  [ "$$(docker inspect ffd-demo --format '{{.State.Health.Status}}' 2>/dev/null)" = "healthy" ] && break; \
	  printf "."; sleep 3; done; echo ""
	@echo "  feed: LIVE STREAM (txn.scored)"

classic:
	@FEED_SOURCE=file $(COMPOSE) up -d demo
	@printf "  waiting for the UI "; for i in $$(seq 1 40); do \
	  [ "$$(docker inspect ffd-demo --format '{{.State.Health.Status}}' 2>/dev/null)" = "healthy" ] && break; \
	  printf "."; sleep 3; done; echo ""
	@echo "  feed: PRE-SCORED FILE (v1 booth demo)"

mode:
	@docker inspect ffd-demo --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null \
	  | grep -E "^FEED_SOURCE=" | sed 's|^|  |' || echo "  demo not running"
	@curl -s --max-time 5 http://localhost:$${DEMO_PORT:-8090}/api/stream 2>/dev/null \
	  | python3 -m json.tool 2>/dev/null | sed 's|^|  |' || true

pipeline: data preprocess train up

clean:
	$(COMPOSE) down --rmi local
