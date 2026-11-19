# Shortcuts for working in this repository. `make` on its own lists them.
#
# Each target runs one command, shown below. On Windows, where make is not installed by default,
# type the command yourself with .venv\Scripts\ in place of .venv/bin/. For example,
# `make run p=fixture` is
#     .venv\Scripts\dantron-etl run fixture
# and `make test-all` is `make db-up`, then
#     $env:DANTRON_TEST_POSTGRES = "postgresql://etl_test:etl_test@localhost:5434/etl_test"
#     .venv\Scripts\pytest tests/ -q

.DEFAULT_GOAL := help
.PHONY: help setup lint test test-all db-up db-down check run view tools catalog docs build

ETL := .venv/bin/dantron-etl
TEST_DB := postgresql://etl_test:etl_test@localhost:5434/etl_test

help: ## list these commands
	@echo "make <command>, where <command> is one of:"
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  %-10s %s\n", $$1, $$2}'

setup: ## install everything this repository uses into .venv
	uv sync --extra dev --extra postgres

lint: ## check the code style
	.venv/bin/ruff check src tests docs/_build

test: ## every test that needs no database
	.venv/bin/pytest tests/ -q

test-all: db-up ## every test, including the Postgres ones (starts the test Postgres first)
	DANTRON_TEST_POSTGRES=$(TEST_DB) .venv/bin/pytest tests/ -q

db-up: ## start the test Postgres (port 5434; its data lives in memory)
	docker compose up -d --wait

db-down: ## stop the test Postgres; its data goes with it
	docker compose down

check: ## check a pipeline file without running it: make check p=pipelines/<name>.yaml
	$(ETL) check $(p)

run: ## preview a pipeline and save the run for the viewer: make run p=<name>
	$(ETL) run $(p)

view: ## open the viewer for this project (ctrl-c to stop)
	$(ETL) view

tools: ## list every tool
	$(ETL) tools

catalog: ## write the catalog of every tool to output_files/
	$(ETL) catalog

docs: ## rewrite the generated pages under docs/ (tools, functions, keys, command help)
	.venv/bin/python docs/_build/generate.py

build: ## build the wheel and the source package into dist/
	uv build
