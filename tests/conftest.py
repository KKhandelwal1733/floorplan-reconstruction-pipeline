"""Auto-generate synthetic fixtures before any test session."""
from pathlib import Path
from tests.fixtures.make_fixtures import make_floor_only, make_with_ceiling

_FIXTURES = Path(__file__).parent / "fixtures"


def pytest_configure(config):
    make_floor_only(_FIXTURES)
    make_with_ceiling(_FIXTURES)
