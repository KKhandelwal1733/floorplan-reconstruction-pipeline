.PHONY: install test lint demo bench ablate

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	python -m py_compile roomscan/cli.py roomscan/schema_out.py roomscan/config.py

demo:
	python -m roomscan.cli tests/fixtures/single_room --out out

bench:
	python -m bench.harness

ablate:
	python -m bench.ablate tests/fixtures/single_room
