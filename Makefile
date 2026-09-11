# Money20/20 Middle East - demo launcher
# Riyadh, 14-16 September 2026
#
#   make <demo>          start a demo (stops any other running demo first)
#   make list            show every demo and whether it is running
#   make stop            stop everything
#
# Adding a demo: drop it in as a subdirectory with its own Makefile exposing
# `up`, `down` and (ideally) `status`, `logs`, `verify`. It is picked up
# automatically - no edit here. Optionally add a one-line `.alias` file to give
# it a short name, e.g. echo fraud > mydemo/.alias

SHELL   := /bin/bash
# Grace period for SIGTERM before Docker escalates to SIGKILL. Ollama and Triton
# need well over the 10s default; a rushed kill is what produced Exit 137.
STOP_TIMEOUT ?= 60

# Which compose project does a demo run as? Declared in .project when it cannot
# be inferred — VSS runs as "mdx" from a compose file three levels down, and the
# document demo runs as "name". Falling back to a compose-file search misses both,
# and a demo the launcher cannot see is a demo it cannot stop.
define proj_of
$$(cat $(1)/.project 2>/dev/null || echo $(1))
endef

# running container count for a demo, via `docker compose ls`
define running_of
$$(docker compose ls --format json 2>/dev/null | python3 -c "import sys,json;p='$(1)';d=json.load(sys.stdin);print(sum(int(x) for e in d if e['Name']==p for x in __import__('re').findall(r'running\((\d+)\)',e['Status'])) or 0)" 2>/dev/null || echo 0)
endef


# A demo's compose file is not always at its root (Synapse keeps it in a
# subdirectory). Resolve it per demo, otherwise `docker compose ps` errors at the
# root, reports nothing running, and the launcher silently fails to stop it —
# leaving the GB10 occupied while another demo starts.
define compose_of
$$(if [ -f "$(1)/docker-compose.yml" ]; then echo "-f $(1)/docker-compose.yml"; \
   elif [ -f "$(1)/compose.yml" ]; then echo "-f $(1)/compose.yml"; \
   else f=$$$$(find "$(1)" -maxdepth 2 \( -name 'docker-compose*.yml' -o -name 'compose*.yml' \) 2>/dev/null | head -1); \
        [ -n "$$$$f" ] && echo "-f $$$$f" || echo ""; fi)
endef
DEMOS   := $(sort $(notdir $(patsubst %/,%,$(dir $(wildcard */Makefile)))))
ALIASES := $(sort $(foreach d,$(DEMOS),$(shell cat $(d)/.alias 2>/dev/null)))

# alias -> directory
define _dir_of
$(strip $(foreach d,$(DEMOS),$(if $(filter $(1),$(d) $(shell cat $(d)/.alias 2>/dev/null)),$(d))))
endef

.DEFAULT_GOAL := help
.PHONY: help list stop stop-others _gpu_free $(DEMOS) $(ALIASES) $(addsuffix -down,$(DEMOS) $(ALIASES)) \
        $(addsuffix -status,$(DEMOS) $(ALIASES)) $(addsuffix -logs,$(DEMOS) $(ALIASES))

help:
	@echo "Money20/20 Middle East - demo launcher"
	@echo
	@echo "  make <demo>           start a demo (stops any other running demo first)"
	@echo "  make <demo>-down      stop it"
	@echo "  make <demo>-status    health, throughput, model quality"
	@echo "  make <demo>-logs      follow its logs"
	@echo "  make <demo>-verify    run its smoke test (if it has one)"
	@echo "  make <demo>-prewarm   warm GPU caches (if it has one)"
	@echo
	@echo "  make list             every demo, and whether it is running"
	@echo "  make stop             stop all demos"
	@echo
	@echo "Demos available:"
	@for d in $(DEMOS); do \
	  a=$$(cat $$d/.alias 2>/dev/null); \
	  if [ -n "$$a" ]; then printf "  %-32s (or: make %s)\n" "make $$d" "$$a"; \
	  else printf "  %-32s\n" "make $$d"; fi; done
	@echo
	@echo "One demo runs at a time. Starting a demo gracefully stops any other"
	@echo "first (SIGTERM, up to $(STOP_TIMEOUT)s to finish) - the GB10 is shared."

list:
	@printf "%-34s %-10s %s\n" DEMO ALIAS STATUS
	@printf "%-34s %-10s %s\n" ------------------------------ --------- --------
	@for d in $(DEMOS); do \
	  a=$$(cat $$d/.alias 2>/dev/null || echo -); \
	  pj=$$(cat $$d/.project 2>/dev/null || echo $$d); \
	  n=$$(docker compose ls --format json 2>/dev/null | python3 -c "import sys,json,re;p='$$pj';d=json.load(sys.stdin);print(next((int(m.group(1)) for e in d if e['Name']==p for m in [re.search(r'running\((\d+)\)',e['Status'])] if m),0))" 2>/dev/null || echo 0); \
	  if [ "$$n" -gt 0 ]; then s="running ($$n)"; else s="stopped"; fi; \
	  printf "%-34s %-10s %s\n" "$$d" "$$a" "$$s"; done

stop:
	@for d in $(DEMOS); do \
	  pj=$$(cat $$d/.project 2>/dev/null || echo $$d); \
	  n=$$(docker compose ls --format json 2>/dev/null | python3 -c "import sys,json,re;p='$$pj';d=json.load(sys.stdin);print(next((int(m.group(1)) for e in d if e['Name']==p for m in [re.search(r'running\((\d+)\)',e['Status'])] if m),0))" 2>/dev/null || echo 0); \
	  if [ "$$n" -gt 0 ]; then printf "  stopping %s gracefully " "$$d"; \
	    $(MAKE) --no-print-directory -C $$d down >/dev/null 2>&1 || true; \
	    for i in $$(seq 1 90); do \
	      left=$$(docker ps -q --filter "label=com.docker.compose.project=$$pj" | wc -l); \
	      [ "$$left" -eq 0 ] && break; printf "."; sleep 1; done; echo " done"; fi; done
	@echo "all demos stopped."

# --- generate per-demo targets for both the directory name and its alias ---
define DEMO_RULES
$(1):
	@for d in $$(DEMOS); do \
	  if [ "$$$$d" != "$(2)" ]; then \
	    pj=$$$$(cat $$$$d/.project 2>/dev/null || echo $$$$d); \
	    n=$$$$(docker ps -q --filter "label=com.docker.compose.project=$$$$pj" | wc -l); \
	    if [ "$$$$n" -gt 0 ]; then \
	      printf "  stopping %s gracefully (shared GB10) " "$$$$d"; \
	      $$(MAKE) --no-print-directory -C $$$$d down >/dev/null 2>&1 || true; \
	      for i in $$$$(seq 1 90); do \
	        left=$$$$(docker ps -q --filter "label=com.docker.compose.project=$$$$pj" | wc -l); \
	        [ "$$$$left" -eq 0 ] && break; printf "."; sleep 1; done; \
	      echo " done"; fi; fi; done
	@$$(MAKE) --no-print-directory _gpu_free
	@echo "starting $(2) ..."
	@$$(MAKE) --no-print-directory -C $(2) up

$(1)-down:
	@$$(MAKE) --no-print-directory -C $(2) down

$(1)-status:
	@$$(MAKE) --no-print-directory -C $(2) status

$(1)-logs:
	@$$(MAKE) --no-print-directory -C $(2) logs

$(1)-verify:
	@$$(MAKE) --no-print-directory -C $(2) verify 2>/dev/null || echo "$(2) has no verify target"

$(1)-prewarm:
	@$$(MAKE) --no-print-directory -C $(2) prewarm 2>/dev/null || echo "$(2) has no prewarm target"
endef

$(foreach d,$(DEMOS),$(eval $(call DEMO_RULES,$(d),$(d))))
$(foreach d,$(DEMOS),$(foreach a,$(shell cat $(d)/.alias 2>/dev/null),$(eval $(call DEMO_RULES,$(a),$(d)))))

# report free GPU + RAM before a demo starts, so contention is visible
_gpu_free:
	@u=$$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | head -1); \
	 m=$$(free -g | awk 'NR==2{print $$7"G free of "$$2"G"}'); \
	 echo "  GB10 ready — gpu util $${u:-n/a} · ram $$m"
