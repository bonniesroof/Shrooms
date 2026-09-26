import pytest

from sim import TICKS_PER_YEAR
from sim.intents import Disturb
from sim.replay import record_run

# One disturbance mid-year so the replay test covers the intent path.
YEAR_INTENTS = [Disturb(tick=4000, x0=8, y0=8, x1=24, y1=24, fraction=0.8)]


@pytest.fixture(scope="session")
def year_run():
    """A full 1-year headless run (8760 ticks), shared across tests."""
    return record_run(seed=42, ticks=TICKS_PER_YEAR, intents=YEAR_INTENTS)
