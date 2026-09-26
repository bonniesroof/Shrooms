import pytest

from sim import TICKS_PER_YEAR
from sim.engine import Simulation
from sim.intents import Disturb, Spill
from sim.params import SimParams, WorldParams
from sim.replay import record_run

# A clearing and a spill mid-year, so the replay test covers both intent kinds.
YEAR_INTENTS = [
    Disturb(tick=4000, x0=8, y0=8, x1=24, y1=24, fraction=0.8),
    Spill(tick=3000, x=45.0, y=20.0, radius=3.0, mass_g=500.0),
]


@pytest.fixture(scope="session")
def year_run():
    """A full 1-year headless run (8760 ticks) on the default 64x64 world, shared."""
    return record_run(seed=42, ticks=TICKS_PER_YEAR, intents=YEAR_INTENTS)


def small_params(**sections) -> SimParams:
    """A 16x16 world for fast unit tests, with optional section overrides."""
    base = SimParams(world=WorldParams(width=16, height=16))
    return base.model_copy(update=sections)


@pytest.fixture
def small_sim():
    return Simulation(7, small_params())
