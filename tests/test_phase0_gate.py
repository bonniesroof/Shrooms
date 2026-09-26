"""Phase 0 'done when': a headless 1-year run completes, replays bit-identically,
and the carbon mass balance holds."""

import numpy as np
import pytest

from sim import TICKS_PER_YEAR
from sim.replay import ReplayMismatch, ReplayRecord, replay
from tests.conftest import YEAR_INTENTS


def test_one_year_run_completes(year_run):
    sim, record = year_run
    assert sim.state.tick == TICKS_PER_YEAR
    for name in ("plant_c", "soil_c", "soil_water"):
        arr = getattr(sim.state, name)
        assert np.isfinite(arr).all(), name
        assert (arr >= 0).all(), name
    assert sim.state.plant_c.mean() > 0, "vegetation collapsed"
    assert record.intents == YEAR_INTENTS


def test_carbon_mass_balance(year_run):
    sim, _ = year_run
    t = sim.ledger.totals
    # Closed system: total C unchanged (also checked every tick during the run).
    assert sim.state.total_carbon() == pytest.approx(sim.ledger.carbon_t0, rel=1e-12)
    # And the atmosphere change is fully explained by booked flows.
    d_atm = sim.state.atmosphere_c - sim.params.atmosphere_initial_c
    explained = t["autotrophic_resp"] + t["heterotrophic_resp"] - t["gpp"]
    assert d_atm == pytest.approx(explained, rel=1e-9, abs=1e-3)
    assert sim.state.land_carbon() == pytest.approx(sim.ledger.expected_land_c(), rel=1e-12)


def test_water_mass_balance(year_run):
    sim, _ = year_run
    assert sim.state.total_water() == pytest.approx(sim.ledger.expected_water(), rel=1e-12)


def test_replay_is_bit_identical(year_run, tmp_path):
    sim, record = year_run
    path = tmp_path / "run.json"
    record.save(path)
    loaded = ReplayRecord.load(path)
    assert loaded == record

    replayed = replay(loaded)
    assert replayed.state.state_hash() == record.final_hash
    for name in ("plant_c", "soil_c", "soil_water"):
        np.testing.assert_array_equal(getattr(replayed.state, name), getattr(sim.state, name))


def test_replay_detects_tampering(year_run):
    _, record = year_run
    tampered = record.model_copy(
        update={"intents": [YEAR_INTENTS[0].model_copy(update={"fraction": 0.5})]}
    )
    with pytest.raises(ReplayMismatch, match="diverged between tick 3650 and 4380"):
        replay(tampered)
