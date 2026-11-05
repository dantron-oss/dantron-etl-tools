# Shortcuts for working in this repository. `make` on its own lists them.
#
# Each one runs a single command shown below. On Windows, where make is not installed by default,
# run that command directly: `make run p=fixture` is
#     uv run python scripts/pipeline_walkthrough_page.py fixture

.DEFAULT_GOAL := help
.PHONY: help setup lint test test-all db-up db-down run view catalog

PY := .venv/bin/python
TEST_DB := postgresql://etl_test:etl_test@localhost:5434/etl_test

# Opens a file in a browser: through Windows from WSL, otherwise with the desktop's own opener.
open = if command -v explorer.exe >/dev/null; then explorer.exe "$$(wslpath -w $(1))" || true; \
       elif command -v xdg-open >/dev/null; then xdg-open $(1); \
       elif command -v open >/dev/null; then open $(1); \
       else echo "open $(1) in a browser"; fi

help: ## list these commands
	@echo "make <command>, where <command> is one of:"
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  %-10s %s\n", $$1, $$2}'

setup: ## install everything this repository uses into .venv
	uv sync --extra dev --extra postgres

lint: ## check the code style
	.venv/bin/ruff check src tests scripts

test: lint ## lint, then every test that needs no database
	.venv/bin/pytest tests/ -q

test-all: lint db-up ## lint, then every test, including the Postgres ones
	DANTRON_TEST_POSTGRES=$(TEST_DB) .venv/bin/pytest tests/ -q

db-up: ## start the test Postgres (port 5434; its data lives in memory)
	docker compose up -d --wait

db-down: ## stop the test Postgres; its data goes with it
	docker compose down

run: ## run a pipeline and open its page: make run p=<name>, for pipelines/<name>.yaml
	@if [ -z "$(p)" ]; then \
	  echo "say which pipeline to run: make run p=<name>"; \
	  echo "the pipelines here are: $$(ls pipelines/*.yaml 2>/dev/null | xargs -n1 basename 2>/dev/null | sed 's/\.yaml$$//' | tr '\n' ' ')"; \
	  exit 2; \
	fi
	$(PY) scripts/pipeline_walkthrough_page.py $(p)

view: ## start the viewer for every saved run, on http://localhost:8899/
	$(PY) scripts/serve_runs.py

catalog: ## rebuild the catalog of every tool and open it
	$(PY) scripts/tool_catalog_page.py
	@$(call open,output_files/tool_catalog.html)
