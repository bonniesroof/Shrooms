"""Core machinery: RNG, determinism, weather, transport, intents, ledger."""

import numpy as np
import pytest

from sim.engine import Simulation
from sim.intents import Disturb, Spill, apply
from sim.ledger import MassBalanceError
from sim.params import SimParams, WeatherParams
from sim.rng import stream
from sim.transport import downslope_drops, fit, route, uniform_fractions
from sim.weather import step_weather
from tests.conftest import small_params


def test_rng_streams_are_seeded_and_independent():
    assert stream(1, "weather").random() == stream(1, "weather").random()
    assert stream(1, "weather").random() != stream(1, "plants").random()
    assert stream(1, "weather").random() != stream(2, "weather").random()


def test_extra_draws_in_one_system_do_not_shift_another():
    s1, s2 = Simulation(3, small_params()), Simulation(3, small_params())
    s2.rng["plants"].random(1000)
    s1.step()
    s2.step()
    assert s1.last_weather == s2.last_weather


def test_hash_catches_rng_drift_before_state_diverges():
    a, b = Simulation(3, small_params()), Simulation(3, small_params())
    b.rng["plants"].random(10)  # drifted stream, no state change yet
    a.run(24)
    b.run(24)
    assert a.state.state_hash() == b.state.state_hash()  # world alone can't tell
    assert a.state_hash() != b.state_hash()


def test_same_seed_same_state_different_seed_differs():
    a, b, c = (Simulation(seed, small_params()) for seed in (5, 5, 6))
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


def test_route_conserves_stacked_fields():
    rng = np.random.default_rng(0)
    x = rng.gamma(2.0, 100.0, size=(3, 16, 16))
    frac = rng.uniform(0, 0.2, size=(4, 16, 16)) * uniform_fractions((16, 16), 1.0) * 4
    y = route(x, frac)
    np.testing.assert_allclose(y.sum(axis=(1, 2)), x.sum(axis=(1, 2)), rtol=1e-13)
    assert (y >= 0).all()


def test_diffusion_keeps_uniform_fields_uniform():
    # Regression: edge cells used to export their full rate over fewer
    # neighbours, draining the borders.
    y = route(np.ones((10, 10)), uniform_fractions((10, 10), 0.3))
    np.testing.assert_allclose(y, 1.0, rtol=1e-14)


def test_downslope_drops_never_point_uphill_or_off_grid():
    e = np.random.default_rng(1).random((12, 12))
    d = downslope_drops(e)
    assert (d >= 0).all()
    assert d[0, 0, :].sum() == d[1, -1, :].sum() == d[2, :, 0].sum() == d[3, :, -1].sum() == 0


def test_fit_scales_demands_to_supply():
    supply = np.array([10.0, 10.0])
    a, b = fit(supply, np.array([8.0, 2.0]), np.array([8.0, 3.0]))
    np.testing.assert_allclose(a + b, [10.0, 5.0])
    np.testing.assert_allclose(a / b, [1.0, 2 / 3])


def test_disturb_moves_all_plant_elements_to_litter(small_sim):
    s = small_sim.state
    before = {el: getattr(s.plant, el)[:4, :4].sum() + getattr(s.litter, el)[:4, :4].sum()
              for el in "cnp"}  # fmt: skip
    apply(Disturb(tick=0, x0=0, y0=0, x1=4, y1=4, fraction=1.0), s, small_sim.ledger)
    assert s.plant.c[:4, :4].max() == 0.0
    for el in "cnp":
        after = getattr(s.litter, el)[:4, :4].sum()
        assert after == pytest.approx(before[el])


def test_spill_adds_exactly_its_mass_and_is_booked(small_sim):
    before = small_sim.state.contaminant.sum()
    small_sim.submit([Spill(tick=0, x=8.0, y=8.0, radius=2.0, mass_g=123.0)])
    small_sim.step()
    assert small_sim.ledger.flows("contaminant")["spill"] == pytest.approx(123.0)
    assert small_sim.accepted and small_sim.state.contaminant.sum() > before + 100


def test_invalid_intents_are_rejected(small_sim):
    small_sim.run(10)
    small_sim.submit(
        [
            Disturb(tick=10, x0=12, y0=0, x1=20, y1=4, fraction=0.5),  # out of bounds
            Spill(tick=10, x=30.0, y=3.0, radius=1.0, mass_g=1.0),  # out of bounds
            Disturb(tick=3, x0=0, y0=0, x1=4, y1=4, fraction=0.5),  # in the past
        ]
    )
    small_sim.step()
    assert not small_sim.accepted
    assert len(small_sim.rejected) == 3


@pytest.mark.parametrize(
    ("substance", "leak"),
    [
        ("c", lambda s: s.som.c.__setitem__((0, 0), s.som.c[0, 0] + 0.01)),
        ("n", lambda s: s.mineral_n.__setitem__((1, 1), s.mineral_n[1, 1] + 1e-4)),
        ("p", lambda s: s.bacteria.p.__setitem__((2, 2), s.bacteria.p[2, 2] + 1e-5)),
        ("water", lambda s: s.water.__setitem__((1, 3, 3), s.water[1, 3, 3] + 0.01)),
        ("contaminant", lambda s: s.contaminant.__setitem__((4, 4), s.contaminant[4, 4] + 0.01)),
    ],
)
def test_ledger_catches_small_leaks(small_sim, substance, leak):
    small_sim.run(5)
    leak(small_sim.state)
    with pytest.raises(MassBalanceError, match=f"unexplained {substance}"):
        small_sim.step()


def test_params_are_frozen_and_strict():
    with pytest.raises(ValueError):
        SimParams(world={"width": 64, "height": 64, "bogus": 1})
