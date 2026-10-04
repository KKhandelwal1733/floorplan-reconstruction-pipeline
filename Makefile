.PHONY: install test lint

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	python -m py_compile roomscan/cli.py roomscan/schema_out.py roomscan/config.py
