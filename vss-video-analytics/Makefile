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

.PHONY: up down start stop restart status ps logs remove redeploy

# Resume if the containers are still there; recreate them from the existing
# generated.env if a teardown removed them. Recreate needs no NGC key and no
# rebuild — every image is already local and the named volumes (model caches,
# postgres, phoenix) survive `down`. A full reset is `redeploy`: destructive,
# needs NGC_CLI_API_KEY, never run implicitly.
up start:
	@cd $(CDIR) && \
	if [ "$$(docker ps -aq --filter label=com.docker.compose.project=$(PROJECT) | wc -l)" -gt 0 ]; then \
	  echo "  resuming existing containers ..."; \
	  $(DC) start; \
	else \
	  test -f "$(ENVFILE)" || { \
	    echo "  ERROR: $(ENVFILE) is missing — this needs a full 'make redeploy'" >&2; \
	    echo "         (destructive: wipes volumes and data-dir, requires NGC_CLI_API_KEY)" >&2; \
	    exit 1; }; \
	  echo "  no containers for project $(PROJECT) — recreating from $(ENVFILE) ..."; \
	  $(DC) up -d; \
	fi
	@printf "  waiting for the agent UI to answer on :3000 "; \
	for i in $$(seq 1 60); do \
	  if [ "$$(curl -s -o /dev/null -w %{http_code} http://localhost:3000/ 2>/dev/null)" = "200" ]; then \
	    echo " ready"; break; fi; \
	  printf "."; sleep 5; \
	done; echo ""
	@echo "  VSS UI : http://localhost:3000  (agent UI)"

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
