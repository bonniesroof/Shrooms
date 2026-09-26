import numpy as np
import pytest

from sim.engine import Simulation
from sim.intents import Disturb
from sim.ledger import MassBalanceError
from sim.params import SimParams, WeatherParams
from sim.plants import disperse
from sim.rng import stream
from sim.weather import step_weather


def test_rng_streams_are_seeded_and_independent():
    a, b = stream(1, "weather"), stream(1, "weather")
    assert a.random() == b.random()
    assert stream(1, "weather").random() != stream(1, "plants").random()
    assert stream(1, "weather").random() != stream(2, "weather").random()


def test_extra_draws_in_one_system_do_not_shift_another():
    s1, s2 = Simulation(3), Simulation(3)
    s2.rng["plants"].random(1000)  # perturb only the plants stream
    s1.step()
    s2.step()
    assert s1.last_weather == s2.last_weather


def test_hash_catches_rng_drift_before_state_diverges():
    a, b = Simulation(3), Simulation(3)
    b.rng["plants"].random(10)  # drifted stream, no state change yet
    a.run(24)
    b.run(24)
    assert a.state.state_hash() == b.state.state_hash()  # world alone can't tell
    assert a.state_hash() != b.state_hash()


def test_same_seed_same_state_different_seed_differs():
    a, b, c = Simulation(5), Simulation(5), Simulation(6)
    for sim in (a, b, c):
        sim.run(200)
    assert a.state_hash() == b.state_hash()
    assert a.state_hash() != c.state_hash()


def test_weather_is_plausible():
    p, rng = WeatherParams(), stream(0, "weather")
    anomaly, raining, rain, temps, night_sw = 0.0, False, 0.0, [], []
    for tick in range(8760):
        wx, anomaly, raining = step_weather(tick, anomaly, raining, p, rng)
        rain += wx.rain_mm
        temps.append(wx.temp_c)
        if wx.hour in (0, 1, 2, 23):
            night_sw.append(wx.shortwave)
    assert 400 < rain < 1400
    assert -25 < min(temps) < 5 < 25 < max(temps) < 40
    assert max(night_sw) == 0.0


def test_dispersal_conserves_carbon():
    rng = np.random.default_rng(0)
    c = rng.gamma(2.0, 100.0, size=(16, 16))
    out = disperse(c, 0.2)
    assert out.sum() == pytest.approx(c.sum(), rel=1e-13)
    assert (out >= 0).all()


def test_disturb_intent_moves_plant_carbon_to_soil():
    sim = Simulation(1)
    plant0, soil0 = sim.state.plant_c.copy(), sim.state.soil_c.copy()
    sim.submit([Disturb(tick=0, x0=0, y0=0, x1=4, y1=4, fraction=1.0)])
    sim.step()
    assert sim.accepted and not sim.rejected
    assert sim.state.plant_c[:4, :4].max() < 1.0
    moved = plant0[:4, :4].sum()
    assert sim.state.soil_c[:4, :4].sum() > soil0[:4, :4].sum() + 0.9 * moved


def test_invalid_intents_are_rejected():
    sim = Simulation(1)
    sim.run(10)
    sim.submit(
        [
            Disturb(tick=10, x0=60, y0=0, x1=70, y1=4, fraction=0.5),  # out of bounds
            Disturb(tick=3, x0=0, y0=0, x1=4, y1=4, fraction=0.5),  # in the past
        ]
    )
    sim.step()
    assert not sim.accepted
    assert len(sim.rejected) == 2


def test_ledger_catches_unexplained_carbon():
    sim = Simulation(1)
    sim.run(5)
    sim.state.soil_c[0, 0] += 1.0  # a leak no flow explains
    with pytest.raises(MassBalanceError, match="unexplained carbon"):
        sim.step()


def test_ledger_catches_tiny_carbon_leak():
    sim = Simulation(1)
    sim.run(5)
    sim.state.plant_c[3, 3] += 0.01  # 10 mg: invisible against the 1e9 g atmosphere
    with pytest.raises(MassBalanceError, match="land"):
        sim.step()


def test_ledger_catches_unexplained_water():
    sim = Simulation(1)
    sim.run(5)
    sim.state.soil_water[0, 0] += 1.0
    with pytest.raises(MassBalanceError, match="unexplained water"):
        sim.step()


def test_params_are_frozen_and_strict():
    with pytest.raises(ValueError):
        SimParams(world={"width": 64, "height": 64, "bogus": 1})
