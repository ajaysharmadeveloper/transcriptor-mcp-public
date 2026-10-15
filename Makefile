# All Python work goes through the repo-root .venv.  Never the system interpreter.
VENV := .venv
PY   := $(VENV)/bin/python
UV   := uv

.DEFAULT_GOAL := help
.PHONY: help install doctor lint test migrate setup clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install:  ## sync dependencies into .venv (add --extra mlx / cuda / cpu for an engine)
	$(UV) sync --all-groups

doctor:  ## platform+engine, ffmpeg, memory pressure, model present, DB reachable, migrations applied
	@$(PY) -m transcriptor.server doctor

lint:  ## enforce the dependency rule (architecture §2, invariant I9)
	@$(VENV)/bin/lint-imports

# The import contracts run first: a dependency-rule violation is architectural,
# so it should fail the build before any behavioural test gets a chance to pass.
test: lint  ## import contracts + pytest (no engine, no model, no GPU needed)
	@$(PY) -m pytest -q

migrate:  ## apply pending migrations
	@$(PY) -m transcriptor.server migrate

setup:  ## download model weights into .transcriptor/models
	@$(PY) -m transcriptor.server setup

bench:  ## rerun the benchmarks and print results for BENCHMARKS.md
	@$(PY) scripts/benchmark.py

build:  ## build a wheel and sdist into dist/
	@$(PY) -m build

clean:  ## remove caches, build output and scratch audio (keeps model weights)
	@rm -rf .pytest_cache .ruff_cache .import_linter_cache dist build
	@find . -path ./.venv -prune -o -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .transcriptor/tmp/*
	@echo "cleaned. model weights in .transcriptor/models are kept — 'rm -rf .transcriptor/models' to drop them too"
