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
DEMOS   := $(sort $(notdir $(patsubst %/,%,$(dir $(wildcard */Makefile)))))
ALIASES := $(sort $(foreach d,$(DEMOS),$(shell cat $(d)/.alias 2>/dev/null)))

# alias -> directory
define _dir_of
$(strip $(foreach d,$(DEMOS),$(if $(filter $(1),$(d) $(shell cat $(d)/.alias 2>/dev/null)),$(d))))
endef

.DEFAULT_GOAL := help
.PHONY: help list stop stop-others $(DEMOS) $(ALIASES) $(addsuffix -down,$(DEMOS) $(ALIASES)) \
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
	@echo "Only one demo runs at a time - they share a single GB10."

list:
	@printf "%-34s %-10s %s\n" DEMO ALIAS STATUS
	@printf "%-34s %-10s %s\n" ------------------------------ --------- --------
	@for d in $(DEMOS); do \
	  a=$$(cat $$d/.alias 2>/dev/null || echo -); \
	  n=$$(cd $$d && docker compose ps -q 2>/dev/null | wc -l); \
	  if [ "$$n" -gt 0 ]; then s="running ($$n)"; else s="stopped"; fi; \
	  printf "%-34s %-10s %s\n" "$$d" "$$a" "$$s"; done

stop:
	@for d in $(DEMOS); do \
	  n=$$(cd $$d && docker compose ps -q 2>/dev/null | wc -l); \
	  if [ "$$n" -gt 0 ]; then echo "stopping $$d ..."; $(MAKE) --no-print-directory -C $$d down >/dev/null 2>&1 || true; fi; done
	@echo "all demos stopped."

# --- generate per-demo targets for both the directory name and its alias ---
define DEMO_RULES
$(1):
	@other=""; for d in $$(DEMOS); do \
	  if [ "$$$$d" != "$(2)" ]; then \
	    n=$$$$(cd $$$$d && docker compose ps -q 2>/dev/null | wc -l); \
	    if [ "$$$$n" -gt 0 ]; then echo "stopping $$$$d (shares the GB10) ..."; \
	      $$(MAKE) --no-print-directory -C $$$$d down >/dev/null 2>&1 || true; fi; fi; done
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
