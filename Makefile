.PHONY: setup install test lint demo bench ablate calibrate repro live

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

# Reproduction bundle (see bench/repro.py docstring): replays plan.json from
# cache/ if present (else computes once and populates it), then runs the
# full live path twice more and checks all three outputs are byte-identical.
repro:
	python -m bench.repro repro tests/fixtures/single_room lidar

# Full live path: always recomputes from raw inputs, cache/ untouched.
live:
	python -m bench.repro live tests/fixtures/single_room lidar
