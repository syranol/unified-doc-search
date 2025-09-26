PYTHON ?= python3
PYTEST ?= $(PYTHON) -m pytest
PIP ?= $(PYTHON) -m pip
RUFF ?= $(PYTHON) -m ruff
REQUIREMENTS_FILE ?= requirements.in

.PHONY: install test lint lint-fix

install:
	$(PIP) install -r $(REQUIREMENTS_FILE)

test:
	$(PYTEST)

lint:
	$(RUFF) check .
	$(RUFF) format --check .

lint-fix:
	$(RUFF) check --fix .
	$(RUFF) format .
