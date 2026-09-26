"""Roadmap gates, checked on one shared default 1-year run.

Phase 0 done-when: a headless 1-year run completes, replays bit-identically,
and the carbon mass balance passes.
Gate A (Phase 1): a stable 1-year run with no default collapse, mass balance
passing for C, N, P and water, and replay identical.
"""

import numpy as np
import pytest

from sim import TICKS_PER_YEAR
from sim.ledger import SUBSTANCES
from sim.params import SimParams
from sim.replay import ReplayMismatch, ReplayRecord, replay
from sim.rng import stream
from sim.world import POOLS, init_world
from tests.conftest import YEAR_INTENTS

# Year-end mean C per cell must stay within these multiples of the starting mean.
COLLAPSE_FLOOR = 0.1
RUNAWAY_CEILING = 10.0


def test_one_year_run_completes(year_run):
    sim, record = year_run
    assert sim.state.tick == TICKS_PER_YEAR
    assert sorted(i.kind for i in record.intents) == sorted(i.kind for i in YEAR_INTENTS)


def test_all_fields_finite_and_non_negative(year_run):
    s = year_run[0].state
    fields = {"water": s.water, "mineral_n": s.mineral_n, "mineral_p": s.mineral_p,
              "contaminant": s.contaminant}  # fmt: skip
    for name in POOLS:
        for el in "cnp":
            fields[f"{name}.{el}"] = getattr(s.pool(name), el)
    for name, arr in fields.items():
        assert np.isfinite(arr).all(), name
        assert arr.min() >= -1e-9, f"{name} went negative: {arr.min()}"


@pytest.mark.parametrize("pool", POOLS)
def test_gate_a_no_collapse_or_runaway(year_run, pool):
    sim = year_run[0]
    start = init_world(SimParams(), stream(42, "init")).pool(pool).c.mean()
    end = sim.state.pool(pool).c.mean()
    assert end >= COLLAPSE_FLOOR * start, f"{pool} collapsed: {start:.2f} -> {end:.2f}"
    assert end <= RUNAWAY_CEILING * start, f"{pool} ran away: {start:.2f} -> {end:.2f}"


def test_mycorrhizal_market_is_active(year_run):
    h = year_run[0].history
    assert sum(h["trade_c"]) > 0 and sum(h["trade_n"]) > 0 and sum(h["trade_p"]) > 0


@pytest.mark.parametrize("substance", SUBSTANCES)
def test_mass_balance(year_run, substance):
    sim = year_run[0]
    got = sim.totals()[substance]
    assert got == pytest.approx(sim.ledger.expected(substance), rel=1e-11, abs=1e-9)


def test_closed_carbon_budget_including_atmosphere(year_run):
    sim = year_run[0]
    now = sim.state.atmosphere_c + sim.totals()["c"]
    assert now == pytest.approx(sim.ledger.atmosphere_c0 + sim.ledger.t0["c"], rel=1e-12)


def test_replay_is_bit_identical(year_run, tmp_path):
    sim, record = year_run
    path = tmp_path / "run.json"
    record.save(path)
    loaded = ReplayRecord.load(path)
    assert loaded == record

    replayed = replay(loaded)
    assert replayed.state_hash() == record.final_hash
    np.testing.assert_array_equal(replayed.state.plant.c, sim.state.plant.c)
    np.testing.assert_array_equal(replayed.state.contaminant, sim.state.contaminant)


def test_replay_detects_tampering(year_run):
    _, record = year_run
    spill = next(i for i in record.intents if i.kind == "spill")
    others = [i for i in record.intents if i is not spill]
    tampered = record.model_copy(
        update={"intents": [*others, spill.model_copy(update={"mass_g": 499.0})]}
    )
    with pytest.raises(ReplayMismatch, match="diverged between tick 2920 and 3650"):
        replay(tampered)
