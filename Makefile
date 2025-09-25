PYTEST ?= pytest
PIP ?= pip
REQUIREMENTS_FILE ?= requirements.in

.PHONY: install test

install:
	$(PIP) install -r $(REQUIREMENTS_FILE)

test:
	$(PYTEST)
