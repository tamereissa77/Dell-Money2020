# ---- Money20/20 launcher compatibility -------------------------------------
# The top-level launcher drives every demo through `up` and `down`.
COMPOSE := docker compose

.PHONY: up down start stop restart status logs ps build

up:
	$(COMPOSE) up -d
	@echo "  Loan Portal : http://localhost:5050"

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

ps status:
	@$(COMPOSE) ps
	@curl -fsS -o /dev/null -w "  portal  : HTTP %{http_code}\n" http://localhost:5050/ 2>/dev/null || echo "  portal  : not responding"

logs:
	$(COMPOSE) logs -f

logs-%:
	$(COMPOSE) logs -f $*
