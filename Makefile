# Shortcuts for working in this repository. `make` on its own lists them.
#
# Each one runs a single command shown below. On Windows, where make is not installed by default,
# run that command directly: `make run p=fixture` is
#     .venv\Scripts\dantron-etl run fixture

.DEFAULT_GOAL := help
.PHONY: help setup lint test test-all db-up db-down run view catalog

ETL := .venv/bin/dantron-etl
TEST_DB := postgresql://etl_test:etl_test@localhost:5434/etl_test

help: ## list these commands
	@echo "make <command>, where <command> is one of:"
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  %-10s %s\n", $$1, $$2}'

setup: ## install everything this repository uses into .venv
	uv sync --extra dev --extra postgres

lint: ## check the code style
	.venv/bin/ruff check src tests

test: lint ## lint, then every test that needs no database
	.venv/bin/pytest tests/ -q

test-all: lint db-up ## lint, then every test, including the Postgres ones
	DANTRON_TEST_POSTGRES=$(TEST_DB) .venv/bin/pytest tests/ -q

db-up: ## start the test Postgres (port 5434; its data lives in memory)
	docker compose up -d --wait

db-down: ## stop the test Postgres; its data goes with it
	docker compose down

run: ## preview a pipeline and save the run for the viewer: make run p=<name>
	$(ETL) run $(p)

view: ## open the viewer for this project (ctrl-c to stop)
	$(ETL) view

catalog: ## write the catalog of every tool to output_files/
	$(ETL) catalog
