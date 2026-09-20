# NVIDIA Video Search & Summarization — wrapper around the existing install.
# The blueprint lives at ~/APPS/VSS-3x (compose project "mdx"); this adapts its
# verbs to the ones the Money20/20 launcher expects.
#
# The launcher enforces one-demo-at-a-time by calling `make down` on whatever is
# running. For VSS that MUST be a pause, not a teardown: the upstream `down`
# runs `compose down --remove-orphans`, which deletes the containers and leaves
# `start` with nothing to resume ("no container found for project mdx"). So
# `down` here maps to upstream `stop` — GPU and RAM are released, containers and
# volumes stay. Use `remove` if you genuinely want the containers gone.

VSS     := $(HOME)/APPS/VSS-3x
CDIR    := $(VSS)/deploy/docker
PROJECT := mdx
PROFILE ?= base
ENVFILE := developer-profiles/dev-profile-$(PROFILE)/generated.env
DC      := docker compose --env-file $(ENVFILE) -p $(PROJECT)

.PHONY: up down start stop restart status ps logs remove redeploy urls

# `compose up -d` covers both cases: containers that exist and are unchanged get
# started, and anything whose config changed gets recreated. That matters here
# because browser-facing URLs (websocket, chat, API) are baked into container
# env at create time — a plain `compose start` would resume them still pointing
# at the previous venue's IP. Volumes and data-dir are untouched either way.
# A full reset is `redeploy`: destructive, needs NGC_CLI_API_KEY, never implicit.
up start:
	@test -f "$(CDIR)/$(ENVFILE)" || { \
	  echo "  ERROR: $(ENVFILE) is missing — this needs a full 'make redeploy'" >&2; \
	  echo "         (destructive: wipes volumes and data-dir, requires NGC_CLI_API_KEY)" >&2; \
	  exit 1; }
	@$(MAKE) --no-print-directory -C $(CDIR) sync-ip PROFILE=$(PROFILE)
	@cd $(CDIR) && $(DC) up -d --remove-orphans
	@printf "  waiting for the agent UI to answer on :3000 "; \
	for i in $$(seq 1 60); do \
	  if [ "$$(curl -s -o /dev/null -w %{http_code} http://localhost:3000/ 2>/dev/null)" = "200" ]; then \
	    echo " ready"; break; fi; \
	  printf "."; sleep 5; \
	done; echo ""
	@echo "  VSS UI : http://$$(tailscale ip -4 2>/dev/null | head -1):3000  (agent UI)"

# Show what the UI will hand the browser — the usual suspect when the venue
# network changes and the UI loads but its websocket will not connect.
urls:
	@$(MAKE) --no-print-directory -C $(CDIR) show-ip PROFILE=$(PROFILE)
	@cd $(CDIR) && $(DC) -f compose.yml config 2>/dev/null \
	  | grep -E "NEXT_PUBLIC.*(WEBSOCKET|CHAT_COMPLETION|API_URL_BASE)" | sort -u | sed 's|^|  |'

# Pause: frees GPU + RAM for the next demo, keeps containers so `up` can resume.
down stop:
	@cd $(CDIR) && $(DC) stop

restart: stop start

ps status:
	@cd $(CDIR) && $(DC) ps
logs:
	@cd $(CDIR) && $(DC) logs -f --tail=200 $(SERVICE)

# Remove the containers (volumes and data-dir survive). `up` will recreate them.
remove:
	@cd $(CDIR) && $(DC) down --remove-orphans

redeploy:
	@$(MAKE) --no-print-directory -C $(VSS) redeploy
