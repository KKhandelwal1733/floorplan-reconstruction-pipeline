.PHONY: setup install test lint demo bench ablate calibrate

setup: install

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	python -m py_compile roomscan/cli.py roomscan/schema_out.py roomscan/config.py

demo:
	python -m roomscan run tests/fixtures/single_room --tier auto --out out

bench:
	python -m bench.harness

ablate:
	python -m bench.ablate tests/fixtures/single_room

calibrate:
	python -m bench.calibrate
