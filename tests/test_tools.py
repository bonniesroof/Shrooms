"""Remediation tools (player) and director events: effects, authority, balances."""

import numpy as np
import pytest

from sim.engine import Simulation
from sim.intents import (
    Amend,
    Downpour,
    Excavate,
    Inoculate,
    Irrigate,
    PestOutbreak,
    Seed,
)
from tests.conftest import small_params


def _apply(sim: Simulation, intent) -> None:
    sim.submit([intent])
    sim.step()  # the engine validates, applies, and checks every balance
    assert sim.accepted and sim.accepted[-1] == intent, sim.rejected


@pytest.mark.parametrize("guild", ["bacteria", "saprotrophs", "mycorrhiza"])
def test_inoculate_adds_exactly_the_guild_carbon(guild):
    sim = Simulation(1, small_params())
    before = sim.state.pool(guild).c.sum()
    _apply(sim, Inoculate(tick=0, x=8, y=8, guild=guild, mass_c_g=100.0))
    assert sim.ledger.flows("c")["inoculation"] == pytest.approx(100.0)
    assert sim.ledger.external_c == pytest.approx(100.0)
    assert sim.state.pool(guild).c.sum() > before + 90.0  # minus one tick of losses


def test_compost_is_nitrogen_rich_litter():
    sim = Simulation(1, small_params())
    _apply(sim, Amend(tick=0, x=8, y=8, mass_c_g=1500.0))
    added_c, added_n = sim.ledger.flows("c")["compost"], sim.ledger.flows("n")["compost"]
    assert added_c / added_n == pytest.approx(15.0)


def test_seed_and_irrigate_are_booked_inflows():
    sim = Simulation(1, small_params())
    _apply(sim, Seed(tick=0, x=4, y=4, mass_c_g=200.0))
    _apply(sim, Irrigate(tick=1, x=4, y=4, water_mm=30.0))
    assert sim.ledger.flows("c")["seeding"] == pytest.approx(200.0)
    assert sim.ledger.flows("water")["irrigation"] > 30.0


def test_excavation_removes_contaminant_and_soil_as_booked_exports():
    sim = Simulation(1, small_params())
    s = sim.state
    s.contaminant[4:8, 4:8] = 10.0
    sim.ledger.t0["contaminant"] = float(s.contaminant.sum())  # re-baseline the edit
    som_before = s.som.c[4:8, 4:8].copy()
    _apply(sim, Excavate(tick=0, x0=4, y0=4, x1=8, y1=8, fraction=0.5))
    assert -sim.ledger.flows("contaminant")["excavation"] == pytest.approx(80.0)
    assert (s.som.c[4:8, 4:8] < 0.51 * som_before).all()
    assert sim.ledger.external_c < 0  # soil carbon left the site, not via the air


def test_director_events_apply():
    sim = Simulation(1, small_params())
    _apply(sim, PestOutbreak(tick=0, agent="director", x=8, y=8, mass_c_g=20.0))
    _apply(sim, Downpour(tick=1, agent="director", x=8, y=8, radius=6, water_mm=50.0))
    assert sim.ledger.flows("water")["storm"] > 50.0


@pytest.mark.parametrize(
    ("intent", "reason"),
    [
        (Seed(tick=0, agent="director", x=4, y=4, mass_c_g=5.0), "authority"),
        (Downpour(tick=0, x=4, y=4, water_mm=5.0), "authority"),  # user can't make weather
        (
            Inoculate(tick=0, agent="mycelium", x=4, y=4, guild="bacteria", mass_c_g=1.0),
            "authority",
        ),  # fmt: skip
        (Irrigate(tick=0, x=40, y=4, water_mm=5.0), "geometry"),
        (Excavate(tick=0, x0=0, y0=0, x1=16, y1=17, fraction=0.5), "geometry"),
    ],
)
def test_tools_are_validated(intent, reason):
    sim = Simulation(1, small_params())
    sim.submit([intent])
    sim.step()
    assert not sim.accepted
    assert sim.rejected and sim.rejected[0][1].startswith(reason)


def test_closed_carbon_budget_holds_with_imports_and_exports():
    sim = Simulation(1, small_params())
    for i, intent in enumerate([
        Amend(tick=0, x=8, y=8, mass_c_g=800.0),
        Excavate(tick=1, x0=0, y0=0, x1=8, y1=8, fraction=0.3),
        Inoculate(tick=2, x=3, y=3, guild="saprotrophs", mass_c_g=50.0),
    ]):  # fmt: skip
        _apply(sim, intent)
        assert sim.state.tick == i + 1
    sim.run(48)
    total = sim.state.atmosphere_c + sim.totals()["c"]
    expected = sim.ledger.atmosphere_c0 + sim.ledger.t0["c"] + sim.ledger.external_c
    assert total == pytest.approx(expected, rel=1e-12)
    assert np.isfinite(sim.state.plant.c).all()
